"""
CLIP ViT-L/14 — Attribute Prediction & Modality Consistency Experiment
========================================================================
Tasks
-----
1. ATTRIBUTE PREDICTION
   Given an image (or caption), rank a set of candidate label strings for
   each target attribute and report top-1 accuracy and mean reciprocal rank
   (MRR).  Three sub-conditions mirror the missing-modality framing:
     • image_only   : image embedding vs. candidate label text embeddings
     • text_only    : caption embedding vs. candidate label text embeddings
     • multimodal   : average of image+caption embeddings vs. candidates

2. MODALITY CONSISTENCY
   For every sample, measure how well the image and caption embeddings
   agree with each other via cosine similarity (I<->T alignment score).
   Additionally compute a rank-consistency metric: for a given attribute,
   does the image and the caption produce the same top-1 prediction?

Models
------
The script uses open_clip's unified API.  Swapping in SigLIP, CyCLIP,
OpenCLIP LAION variants, or any other open_clip-compatible checkpoint
requires only editing the MODELS registry at the bottom of this file
(or passing --model / --pretrained CLI flags).

Default: openai / ViT-L-14  (the standard CLIP ViT-L/14 checkpoint)

Expected DataFrame schema
-------------------------
Column names are configurable via the COL_* constants below.

  image_path : str  — path to the product image file
  caption    : str  — free-text product description / marketing copy
  labels     : dict or JSON str — ground-truth attributes, e.g.
               {"flavors": "mango", "color": "blue", "item": "disposable vape"}

Candidate label sets
--------------------
For zero-shot attribute prediction, CLIP needs a finite set of candidate
strings per attribute.  You can supply these in three ways (in priority order):
  1. Pass --candidates path/to/candidates.json  (a dict mapping attr → [str])
  2. Define them inline in CANDIDATE_LABELS below
  3. Auto-derive from the unique values observed in the DataFrame labels column
     (set AUTO_CANDIDATES = True)

Usage
-----
  python clip_attribute_consistency.py \
      --data   path/to/dataframe.pkl   \
      --out    results/                \
      --model  ViT-L-14                \
      --pretrained openai              \
      --sample 300                     \
      --candidates candidates.json

Dependencies
------------
  pip install open_clip_torch torch pillow pandas tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from tqdm import tqdm

img_file_root = "/home/serna/Programming/smart_connect_health_neurips_2026/data"

# ---------------------------------------------------------------------------
# Column name configuration — edit to match your DataFrame
# ---------------------------------------------------------------------------
COL_IMAGE_PATH = "filepath"
COL_CAPTION    = "description"
COL_LABELS     = "product_name"

REQUIRED_COLS  = [COL_IMAGE_PATH, COL_CAPTION, COL_LABELS]

# Attributes to evaluate — must match keys present in the labels dicts
TARGET_ATTRIBUTES = ["item", "flavors", "color"] + ["marketing", "shape", "text", "product_name"]

# ---------------------------------------------------------------------------
# Candidate label sets for zero-shot attribute prediction
# ---------------------------------------------------------------------------
# Fill these in with the label vocabulary from your dataset, or set
# AUTO_CANDIDATES = True to derive them automatically from the DataFrame.

AUTO_CANDIDATES = True   # set False and populate the dict below for full control

CANDIDATE_LABELS: dict[str, list[str]] = {
    # "flavors": ["mango", "watermelon", "mint", "tobacco", "menthol", ...],
    # "color":  ["blue", "red", "green", "black", "white", "purple", ...],
    # "item":   ["disposable vape", "cigarette", "nicotine pouch", ...],
}

# Prompt templates — CLIP benefits from natural-language wrapping
PROMPT_TEMPLATES: dict[str, str] = {
    "item":   "a photo of a {}",
    "flavors": "a {} flavored product",
    "color":  "an object that is {} in color",
    "marketing": "a photo of an item for {}",
    "shape": "an object with {} shape",
    "text": "a photo with text: {}",
    "product_name": "a photo of a {}",
    "_default": "a photo of a {}",
}

# ---------------------------------------------------------------------------
# Model registry — add new models here to run multi-model comparisons
# ---------------------------------------------------------------------------
# Each entry: (open_clip model_name, pretrained_tag)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "clip_vitl14_openai":  ("ViT-L-14",  "openai"),
    "clip_vitb16_openai":  ("ViT-B-16", "openai"),
    # "clip_vitl14_laion2b": ("ViT-L-14",  "laion2b_s32b_b82k"),  # OpenCLIP LAION variant
    # "siglip_vitl14":       ("ViT-L-14-quickgelu", "dfn2b"),       # SigLIP-style
    # "cyclip_vitb32":       ("ViT-B-32",  "laion400m_e32"),        # CyCLIP (use its ckpt)
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ===========================================================================
# Model wrapper — single interface for all open_clip models
# ===========================================================================

class CLIPWrapper:
    """
    Thin wrapper around an open_clip model that exposes:
      - encode_image(pil_image)  → normalised float32 numpy vector
      - encode_text(str_list)    → normalised float32 numpy array  (N, D)
    """

    def __init__(self, model_name: str, pretrained: str, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        log.info(f"Loading open_clip model={model_name!r} pretrained={pretrained!r} "
                 f"on device={self.device!r}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()
        self.model_name  = model_name
        self.pretrained  = pretrained

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Return an L2-normalised image embedding as a 1-D numpy array."""
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feat   = self.model.encode_image(tensor)
        feat   = F.normalize(feat, dim=-1)
        return feat.cpu().float().numpy()[0]   # shape (D,)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Return an L2-normalised text embedding matrix as a 2-D numpy array."""
        tokens = self.tokenizer(texts).to(self.device)
        feats  = self.model.encode_text(tokens)
        feats  = F.normalize(feats, dim=-1)
        return feats.cpu().float().numpy()     # shape (N, D)


def build_wrapper(model_key: str) -> CLIPWrapper:
    model_name, pretrained = MODEL_REGISTRY[model_key]
    return CLIPWrapper(model_name, pretrained)


# ===========================================================================
# Candidate label helpers
# ===========================================================================

def apply_template(label: str, attribute: str) -> str:
    template = PROMPT_TEMPLATES.get(attribute, PROMPT_TEMPLATES["_default"])
    return template.format(label)


def derive_candidates(df: pd.DataFrame, attributes: list[str]) -> dict[str, list[str]]:
    """Build candidate sets from unique ground-truth values in the DataFrame.

    NOTE: When deriving candidate sets of labels + attributes, assume the attributes are in their
    own columns in the dataframe df.
    """
    candidates: dict[str, list[str]] = defaultdict(set)

    for attribute in attributes:
        unique_values = df[attribute].unique()
        for uv in unique_values:
            candidates[attribute].add(uv.strip().lower())

        #candidates[attribute].add(unique_values)

    # TODO: find out what candidates is for, is it all the possible values for columns/labels we want to predict? it seems so
    #  comment out the below loop and do the attribute: [set of possible values] myself

    '''for labels in df[COL_LABELS]: # iterate over each sample
        for attr in attributes: # iterate over each sample's attributes of interest

            candidates[attr].add(str(attr).strip().lower())
            #val = labels.get(attr)
            #if val is not None:
            #    candidates[attr].add(str(val).strip().lower())'''
    return {attr: sorted(vals) for attr, vals in candidates.items()}


def load_candidates(path: str) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# Task 1 — Attribute Prediction
# ===========================================================================

def predict_attribute(
    query_embedding: np.ndarray,           # (D,)
    candidate_embeddings: np.ndarray,      # (K, D)  pre-encoded
    candidate_labels: list[str],
) -> tuple[str, list[tuple[str, float]]]:
    """
    Rank candidates by cosine similarity and return:
      - top-1 predicted label
      - full ranked list of (label, score) tuples
    """
    scores  = candidate_embeddings @ query_embedding   # (K,)
    ranking = sorted(zip(candidate_labels, scores.tolist()), key=lambda x: -x[1])
    return ranking[0][0], ranking


def reciprocal_rank(gt: str, ranking: list[tuple[str, float]]) -> float:
    for rank, (label, _) in enumerate(ranking, start=1):
        if label.lower().strip() == gt.lower().strip():
            return 1.0 / rank
    return 0.0


# ===========================================================================
# Task 2 — Modality Consistency
# ===========================================================================

def image_text_cosine(img_emb: np.ndarray, txt_emb: np.ndarray) -> float:
    """Cosine similarity between a single image and text embedding."""
    # Both are already L2-normalised
    return float(np.dot(img_emb, txt_emb))


def rank_consistency(
    img_pred: str,
    txt_pred: str,
) -> bool:
    """True if both modalities produce the same top-1 attribute prediction."""
    return img_pred.lower().strip() == txt_pred.lower().strip()


# ===========================================================================
# Pre-encode candidate labels (once per attribute, reused across samples)
# ===========================================================================

def precompute_candidate_embeddings(
    wrapper: CLIPWrapper,
    candidates: dict[str, list[str]],
) -> dict[str, tuple[list[str], np.ndarray]]:
    """
    Returns {attribute: (label_list, embedding_matrix)} where
    embedding_matrix is shape (K, D), L2-normalised.
    """
    result = {}

    for attr, labels in candidates.items():
        prompted = [apply_template(l, attr) for l in labels]
        embs     = wrapper.encode_texts(prompted)           # (K, D)
        result[attr] = (labels, embs)
        log.info(f"  Pre-encoded {len(labels)} candidates for attribute '{attr}'")
    return result


# ===========================================================================
# Main experiment loop
# ===========================================================================

def run_experiment(
    df: pd.DataFrame,
    wrapper: CLIPWrapper,
    candidates: dict[str, list[str]],
    attributes: list[str],
    out_dir: Path,
    model_key: str,
) -> pd.DataFrame:
    """
    Run attribute prediction (3 conditions) and modality consistency over df.
    """
    log.info(f"Pre-computing candidate embeddings for {list(candidates.keys())}")
    cand_store = precompute_candidate_embeddings(wrapper, candidates)

    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_key}"):
        # ---- Load image ----
        try:
            image = Image.open(row[COL_IMAGE_PATH]).convert("RGB")
            img_emb = wrapper.encode_image(image)
            img_ok  = True
        except Exception as e:
            log.warning(f"[{idx}] Image load failed: {e}")
            img_emb = np.zeros(512, dtype=np.float32)   # fallback zero vector
            img_ok  = False

        # ---- Encode caption ----
        caption = str(row[COL_CAPTION])
        txt_emb = wrapper.encode_texts([caption])[0]    # (D,)

        # ---- Fused (multimodal) embedding ---- avg then renorm
        fused_emb = img_emb + txt_emb
        norm      = np.linalg.norm(fused_emb)
        fused_emb = fused_emb / norm if norm > 1e-8 else fused_emb

        ground_truth: dict[str, str] = {}
        #ground_truth = str(row[COL_LABELS])
        for attr in TARGET_ATTRIBUTES:
            ground_truth[attr] = row[attr]

        # ---- Modality consistency: I<->T cosine ----
        it_cosine = image_text_cosine(img_emb, txt_emb)

        record: dict[str, Any] = {
            "sample_index":    idx,
            "model":           model_key,
            "image_ok":        img_ok,
            "it_cosine":       round(it_cosine, 6),
            "ground_truth":    ground_truth#json.dumps(ground_truth),
        }

        # ---- Per-attribute scoring ----
        for attr in attributes:
            if attr not in cand_store:
                continue
            label_list, cand_embs = cand_store[attr]
            gt_val = str(ground_truth.get(attr, "")).lower().strip()

            # -- image_only condition --
            img_pred, img_ranking = predict_attribute(img_emb, cand_embs, label_list)
            #img_hit = int(img_pred.lower() == gt_val)
            if img_pred.lower() in gt_val.lower():
                img_hit = 1
            elif gt_val.lower() in img_pred.lower():
                img_hit = 1
            else:
                img_hit = 0
            img_mrr = reciprocal_rank(gt_val, img_ranking)

            # -- text_only condition --
            txt_pred, txt_ranking = predict_attribute(txt_emb, cand_embs, label_list)
            #txt_hit = int(txt_pred.lower() == gt_val)
            if txt_pred.lower() in gt_val.lower():
                txt_hit = 1
            elif gt_val.lower() in txt_pred.lower():
                txt_hit = 1
            else:
                txt_hit = 0
            txt_mrr = reciprocal_rank(gt_val, txt_ranking)

            # -- multimodal (fused) condition --
            fus_pred, fus_ranking = predict_attribute(fused_emb, cand_embs, label_list)
            #fus_hit = int(fus_pred.lower() == gt_val)
            if fus_pred.lower() in gt_val.lower():
                fus_hit = 1
            elif gt_val.lower() in fus_pred.lower():
                fus_hit = 1
            else:
                fus_hit = 0
            fus_mrr = reciprocal_rank(gt_val, fus_ranking)

            # -- rank consistency (do image & text agree?) --
            consistent = int(rank_consistency(img_pred, txt_pred))

            record.update({
                "filepath":                 row.filepath,
                f"{attr}_gt":               gt_val,
                # image_only
                f"{attr}_img_pred":         img_pred,
                f"{attr}_img_hit":          img_hit,
                f"{attr}_img_mrr":          round(img_mrr, 6),
                # text_only
                f"{attr}_txt_pred":         txt_pred,
                f"{attr}_txt_hit":          txt_hit,
                f"{attr}_txt_mrr":          round(txt_mrr, 6),
                # multimodal
                f"{attr}_fused_pred":       fus_pred,
                f"{attr}_fused_hit":        fus_hit,
                f"{attr}_fused_mrr":        round(fus_mrr, 6),
                # consistency
                f"{attr}_rank_consistent":  consistent,
            })

        records.append(record)

    results_df = pd.DataFrame(records)

    # ---- Save per-sample results ----
    out_csv = out_dir / f"{model_key}_results.csv"
    results_df.to_csv(out_csv, index=False)
    log.info(f"Per-sample results saved → {out_csv}")

    # ---- Aggregate summary ----
    summary = _compute_summary(results_df, attributes)
    out_json = out_dir / f"{model_key}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved → {out_json}")

    _print_summary(summary, model_key)
    return results_df


# ===========================================================================
# Summary computation
# ===========================================================================

def _compute_summary(df: pd.DataFrame, attributes: list[str]) -> dict:
    """
    Aggregate per-sample results into:
      - per-attribute accuracy and MRR for each condition
      - overall (macro-average) accuracy and MRR per condition
      - mean I<->T cosine similarity
      - per-attribute rank-consistency rate
    """
    conditions = ["img", "txt", "fused"]
    summary: dict[str, Any] = {
        "n_samples":        int(len(df)),
        "n_valid_images":   int(df["image_ok"].sum()),
        "mean_it_cosine":   round(float(df["it_cosine"].mean()), 4),
        "attributes":       {},
        "overall":          {},
    }

    all_hits   = {c: [] for c in conditions}
    all_mrrs   = {c: [] for c in conditions}

    for attr in attributes:
        attr_entry = {}
        for cond in conditions:
            hit_col = f"{attr}_{cond}_hit"
            mrr_col = f"{attr}_{cond}_mrr"
            if hit_col not in df.columns:
                continue
            hits = df[hit_col].dropna()
            mrrs = df[mrr_col].dropna()
            acc  = round(float(hits.mean()), 4)
            mrr  = round(float(mrrs.mean()), 4)
            attr_entry[cond] = {"accuracy": acc, "mrr": mrr}
            all_hits[cond].extend(hits.tolist())
            all_mrrs[cond].extend(mrrs.tolist())

        cons_col = f"{attr}_rank_consistent"
        if cons_col in df.columns:
            attr_entry["rank_consistency"] = round(
                float(df[cons_col].dropna().mean()), 4
            )

        summary["attributes"][attr] = attr_entry

    for cond in conditions:
        if all_hits[cond]:
            summary["overall"][cond] = {
                "accuracy": round(float(np.mean(all_hits[cond])), 4),
                "mrr":      round(float(np.mean(all_mrrs[cond])), 4),
            }

    # Overall rank consistency (macro-average across attributes)
    cons_vals = [
        summary["attributes"][a]["rank_consistency"]
        for a in attributes
        if "rank_consistency" in summary["attributes"].get(a, {})
    ]
    if cons_vals:
        summary["overall"]["mean_rank_consistency"] = round(float(np.mean(cons_vals)), 4)

    return summary


def _print_summary(summary: dict, model_key: str) -> None:
    print("\n" + "=" * 70)
    print(f"RESULTS — {model_key}")
    print("=" * 70)
    print(f"  Samples          : {summary['n_samples']}")
    print(f"  Valid images     : {summary['n_valid_images']}")
    print(f"  Mean I<->T cosine  : {summary['mean_it_cosine']}")
    print()

    header = f"  {'Attribute':<14} {'Condition':<10} {'Accuracy':>10} {'MRR':>10} {'RankConsist':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for attr, entry in summary["attributes"].items():
        cons = entry.get("rank_consistency", float("nan"))
        first = True
        for cond in ["img", "txt", "fused"]:
            if cond not in entry:
                continue
            acc = entry[cond]["accuracy"]
            mrr = entry[cond]["mrr"]
            attr_label = attr if first else ""
            cons_str   = f"{cons:.4f}" if first else ""
            print(f"  {attr_label:<14} {cond:<10} {acc:>10.4f} {mrr:>10.4f} {cons_str:>13}")
            first = False
        print()

    print("  OVERALL")
    for cond in ["img", "txt", "fused"]:
        if cond in summary["overall"]:
            o = summary["overall"][cond]
            print(f"    {cond:<10} acc={o['accuracy']:.4f}  mrr={o['mrr']:.4f}")
    if "mean_rank_consistency" in summary["overall"]:
        print(f"    mean rank consistency = {summary['overall']['mean_rank_consistency']:.4f}")
    print("=" * 70 + "\n")


# ===========================================================================
# DataFrame loading
# ===========================================================================

def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    loaders = {".pkl": pd.read_pickle, ".csv": pd.read_csv,
               ".parquet": pd.read_parquet, ".pq": pd.read_parquet}
    loader  = loaders.get(p.suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    df = loader(p, index_col=0)
    df[COL_IMAGE_PATH] = img_file_root + '/' + df[COL_IMAGE_PATH].astype(str)

    debug = True
    if debug:
        print(f"[DEBUG] REMOVING NaN VALUES FROM DATAFRAME!!! Current samples: {len(df)}")
        df.dropna(subset=[COL_LABELS], inplace=True)
        df.dropna(subset=TARGET_ATTRIBUTES, inplace=True)
        print(f"[DEBUG] samples: {len(df)}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Parse labels if stored as JSON string
    #if df[COL_LABELS].dtype == object and isinstance(df[COL_LABELS].iloc[0], str):
    #    df[COL_LABELS] = df[COL_LABELS].apply(json.loads)

    # Normalize string fields
    df[COL_CAPTION]    = df[COL_CAPTION].astype(str).str.strip()
    df[COL_IMAGE_PATH] = df[COL_IMAGE_PATH].astype(str).str.strip()

    return df


# ===========================================================================
# CLI entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLIP attribute prediction & modality consistency pilot"
    )
    parser.add_argument("--data",        required=True,
                        help="Path to input DataFrame (.pkl / .csv / .parquet)")
    parser.add_argument("--out",         default="results",
                        help="Output directory")
    parser.add_argument("--model",       default=None, #default="ViT-L-14",
                        required=True,
                        help="open_clip model name (e.g., ViT-L-14)") # ViT-B-16
    parser.add_argument("--pretrained",  default="openai",
                        help="open_clip pretrained tag (default: openai)")
    parser.add_argument("--model-key",   default=None, #default="clip_vitl14_openai",
                        required=True,
                        help="Label key for this run used in output filenames")
    parser.add_argument("--candidates",  default=None,
                        help="Path to JSON file mapping attribute → [candidate labels]")
    parser.add_argument("--sample",      type=int, default=None,
                        help="Subsample N rows for piloting (None = all rows)")
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load data --
    log.info(f"Loading DataFrame from {args.data}")
    df = load_dataframe(args.data)
    log.info(f"Loaded {len(df)} rows")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        log.info(f"Subsampled to {len(df)} rows")

    # -- Candidates --
    if args.candidates:
        log.info(f"Loading candidate labels from {args.candidates}")
        candidates = load_candidates(args.candidates)
    elif CANDIDATE_LABELS and not AUTO_CANDIDATES:
        candidates = CANDIDATE_LABELS
    else:
        log.info("Auto-deriving candidate labels from DataFrame ground-truth values")
        candidates = derive_candidates(df, TARGET_ATTRIBUTES)
        log.info(f"Derived candidates: { {k: len(v) for k, v in candidates.items()} }")

    # -- Override MODEL_REGISTRY entry if CLI flags differ from default --
    model_key = args.model_key
    MODEL_REGISTRY[model_key] = (args.model, args.pretrained)

    # -- Load model --
    wrapper = build_wrapper(model_key)

    # -- Run --
    run_experiment(
        df=df,
        wrapper=wrapper,
        candidates=candidates,
        attributes=TARGET_ATTRIBUTES,
        out_dir=out_dir,
        model_key=model_key,
    )


if __name__ == "__main__":
    main()
