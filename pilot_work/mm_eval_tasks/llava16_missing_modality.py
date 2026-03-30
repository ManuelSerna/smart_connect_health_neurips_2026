"""
LLaVA-1.6 (LLaVA-NeXT) Missing Modality Experiment - Pilot Script
====================================================================
Tasks:
  1. image_only      : predict attributes from image alone (no caption/text)
  2. text_only       : predict attributes from caption alone (no image)
  3. full_multimodal : predict attributes from image + caption together

Each condition is run over every row in a pandas DataFrame that contains at
minimum the columns listed in REQUIRED_COLS. Results are written to a CSV and
a JSON summary with per-attribute and per-condition accuracy metrics.

Expected DataFrame schema (column names are configurable via constants below):
  - image_path  : str  - absolute or relative path to the product image
  - caption     : str  - free-text product description
  - labels      : dict or str - ground-truth attribute dict, e.g.
                  {"flavor": "mango", "color": "blue", "item": "disposable vape"}
                  If stored as a string, it will be json.loads()'d automatically.

Usage:
  python llava16_missing_modality.py \
      --data  path/to/dataframe.pkl \   # or .csv - see load_dataframe()
      --out   results/                  \
      --model llava-hf/llava-v1.6-mistral-7b-hf \
      --sample 200                      # optional: subsample N rows for piloting

Dependencies:
  pip install torch transformers pillow pandas tqdm accelerate bitsandbytes
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)

# ---------------------------------------------------------------------------
# Configuration - edit these to match your DataFrame's actual column names
# ---------------------------------------------------------------------------
COL_IMAGE_PATH = "image_path"
COL_CAPTION    = "caption"
COL_LABELS     = "labels"          # dict with keys like "flavor", "color", etc.

REQUIRED_COLS  = [COL_IMAGE_PATH, COL_CAPTION, COL_LABELS]

# Attributes you want to evaluate; must be keys present in the labels dict
TARGET_ATTRIBUTES = ["item", "flavor", "color"]

# Placeholder image used for text-only condition (blank white 336x336)
BLANK_IMAGE_SIZE = (336, 336)

# Generation settings
MAX_NEW_TOKENS = 128
TEMPERATURE    = 0.0        # greedy decoding for reproducibility

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates for each condition
# ---------------------------------------------------------------------------

def build_prompt_image_only(attributes: list[str]) -> list[dict]:
    """Image is provided; no caption. Ask model to predict each attribute."""
    attr_list = ", ".join(attributes)
    text = (
        f"You are a product analysis assistant. "
        f"Look at the product image and predict the following attributes: {attr_list}. "
        f"Respond ONLY with a JSON object where keys are the attribute names and "
        f"values are your predictions. Do not include explanations."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def build_prompt_text_only(caption: str, attributes: list[str]) -> list[dict]:
    """
    Text-only condition. We still pass a blank image because LLaVA-1.6 requires
    an image token in its prompt. The blank image carries no meaningful signal.
    """
    attr_list = ", ".join(attributes)
    text = (
        f"You are a product analysis assistant. "
        f"Based ONLY on the following product description (ignore the image), "
        f"predict the following attributes: {attr_list}.\n\n"
        f"Product description: {caption}\n\n"
        f"Respond ONLY with a JSON object where keys are the attribute names and "
        f"values are your predictions. Do not include explanations."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},   # blank image - model is told to ignore it
                {"type": "text", "text": text},
            ],
        }
    ]


def build_prompt_full_multimodal(caption: str, attributes: list[str]) -> list[dict]:
    """Both image and caption are available."""
    attr_list = ", ".join(attributes)
    text = (
        f"You are a product analysis assistant. "
        f"Using BOTH the product image and the following description, "
        f"predict the following attributes: {attr_list}.\n\n"
        f"Product description: {caption}\n\n"
        f"Respond ONLY with a JSON object where keys are the attribute names and "
        f"values are your predictions. Do not include explanations."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, use_4bit: bool = True):
    """
    Load LLaVA-1.6 with optional 4-bit quantization (recommended for <= 24 GB VRAM).
    Set use_4bit=False if you have ample GPU memory or are on CPU (slow).
    """
    log.info(f"Loading processor from {model_id}")
    processor = LlavaNextProcessor.from_pretrained(model_id)

    quant_config = None
    if use_4bit and torch.cuda.is_available():
        log.info("Using 4-bit quantization (BitsAndBytes NF4)")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    log.info(f"Loading model from {model_id}")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    log.info("Model loaded successfully")
    return processor, model


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def run_inference(
    processor,
    model,
    conversation: list[dict],
    image: Image.Image,
) -> str:
    """
    Apply chat template, tokenize, generate, and decode the response.
    Returns only the newly generated tokens (strips the input prompt).
    """
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,          # greedy - deterministic for benchmarking
            temperature=None,
            top_p=None,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict[str, str]:
    """
    Extract a JSON object from the model's raw output text.
    Handles cases where the model wraps JSON in markdown code fences.
    Returns an empty dict on failure.
    """
    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Try to find the first {...} block
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: try parsing the whole string
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning(f"Could not parse JSON from model output: {repr(raw)}")
        return {}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def normalize(value: Any) -> str:
    """Lowercase and strip whitespace for fuzzy string matching."""
    return str(value).lower().strip()


def compute_attribute_accuracy(
    predictions: list[dict],
    ground_truths: list[dict],
    attributes: list[str],
) -> dict[str, float]:
    """
    Compute per-attribute exact-match accuracy (case-insensitive).

    Args:
        predictions  : list of predicted attribute dicts, one per sample
        ground_truths: list of ground-truth attribute dicts, one per sample
        attributes   : list of attribute names to evaluate

    Returns:
        dict mapping attribute name → accuracy in [0, 1]
    """
    counts  = {a: 0 for a in attributes}
    correct = {a: 0 for a in attributes}

    for pred, gt in zip(predictions, ground_truths):
        for attr in attributes:
            gt_val   = gt.get(attr)
            pred_val = pred.get(attr)
            if gt_val is None:
                continue                          # skip missing ground truth
            counts[attr] += 1
            if pred_val is not None and normalize(pred_val) == normalize(gt_val):
                correct[attr] += 1

    return {
        attr: (correct[attr] / counts[attr] if counts[attr] > 0 else float("nan"))
        for attr in attributes
    }


# ---------------------------------------------------------------------------
# DataFrame loading
# ---------------------------------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    """Load a DataFrame from .pkl, .csv, or .parquet."""
    p = Path(path)
    if p.suffix == ".pkl":
        df = pd.read_pickle(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Parse labels column if stored as string
    if df[COL_LABELS].dtype == object and isinstance(df[COL_LABELS].iloc[0], str):
        df[COL_LABELS] = df[COL_LABELS].apply(json.loads)

    return df


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

BLANK_IMAGE = Image.new("RGB", BLANK_IMAGE_SIZE, color=(255, 255, 255))

CONDITIONS = ["image_only", "text_only", "full_multimodal"]


def run_experiment(
    df: pd.DataFrame,
    processor,
    model,
    attributes: list[str],
    out_dir: Path,
) -> pd.DataFrame:
    """
    Run all three missing-modality conditions over the DataFrame.

    Returns a results DataFrame with one row per (sample, condition) with
    columns: [index, condition, ground_truth, prediction, per-attribute hits].
    """
    records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Samples"):
        # --- Load image ---
        try:
            image = Image.open(row[COL_IMAGE_PATH]).convert("RGB")
        except Exception as e:
            log.warning(f"[{idx}] Could not load image {row[COL_IMAGE_PATH]}: {e}")
            image = BLANK_IMAGE     # fall back; marks this sample as degraded

        caption     = str(row[COL_CAPTION])
        ground_truth = row[COL_LABELS]   # dict

        for condition in CONDITIONS:
            # Build prompt
            if condition == "image_only":
                conversation = build_prompt_image_only(attributes)
                img_input    = image
            elif condition == "text_only":
                conversation = build_prompt_text_only(caption, attributes)
                img_input    = BLANK_IMAGE      # blank - no visual signal
            else:  # full_multimodal
                conversation = build_prompt_full_multimodal(caption, attributes)
                img_input    = image

            # Inference
            try:
                raw_output = run_inference(processor, model, conversation, img_input)
            except Exception as e:
                log.error(f"[{idx}] Inference failed for condition={condition}: {e}")
                raw_output = "{}"

            prediction = parse_json_response(raw_output)

            # Per-attribute hit flags
            attr_hits = {}
            for attr in attributes:
                gt_val   = ground_truth.get(attr)
                pred_val = prediction.get(attr)
                if gt_val is None:
                    attr_hits[f"hit_{attr}"] = None
                else:
                    attr_hits[f"hit_{attr}"] = (
                        pred_val is not None
                        and normalize(pred_val) == normalize(gt_val)
                    )

            records.append(
                {
                    "sample_index": idx,
                    "condition":    condition,
                    "ground_truth": json.dumps(ground_truth),
                    "raw_output":   raw_output,
                    "prediction":   json.dumps(prediction),
                    **attr_hits,
                }
            )

    results_df = pd.DataFrame(records)

    # --- Save per-sample results ---
    out_csv = out_dir / "missing_modality_results.csv"
    results_df.to_csv(out_csv, index=False)
    log.info(f"Per-sample results saved to {out_csv}")

    # --- Compute and save summary ---
    summary = {}
    for condition in CONDITIONS:
        subset = results_df[results_df["condition"] == condition]
        cond_summary = {}
        for attr in attributes:
            col = f"hit_{attr}"
            valid = subset[col].dropna()
            cond_summary[attr] = round(float(valid.mean()), 4) if len(valid) > 0 else None
        # Overall accuracy across all attributes
        hit_cols = [f"hit_{a}" for a in attributes]
        all_hits = subset[hit_cols].values.flatten()
        all_hits = all_hits[all_hits != None].astype(float)  # noqa: E711
        cond_summary["overall"] = round(float(all_hits.mean()), 4) if len(all_hits) > 0 else None
        summary[condition] = cond_summary

    out_json = out_dir / "missing_modality_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {out_json}")

    # Pretty-print summary to console
    print("\n" + "=" * 60)
    print("MISSING MODALITY EXPERIMENT - RESULTS SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(summary).T
    print(summary_df.to_string())
    print("=" * 60 + "\n")

    return results_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLaVA-1.6 missing modality pilot experiment"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the input DataFrame (.pkl / .csv / .parquet)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results",
        help="Output directory for CSVs and JSON summary",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-v1.6-mistral-7b-hf",
        help="HuggingFace model ID for LLaVA-NeXT",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly subsample N rows (useful for piloting); None = use all",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use full fp16 - requires more VRAM)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    log.info(f"Loading data from {args.data}")
    df = load_dataframe(args.data)
    log.info(f"Loaded {len(df)} rows")

    if args.sample is not None and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).reset_index(drop=True)
        log.info(f"Subsampled to {len(df)} rows")

    # Load model
    processor, model = load_model(args.model, use_4bit=not args.no_4bit)

    # Run experiment
    run_experiment(
        df=df,
        processor=processor,
        model=model,
        attributes=TARGET_ATTRIBUTES,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()