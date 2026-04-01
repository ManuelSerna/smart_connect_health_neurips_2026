import argparse
import json
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm


# Global vars
DELTA = 32 # number of samples per batch, how many idxs to skip in df

MAX_IM_SIDE_LEN = 1000
MAX_ASSIGN_LEN = 854 # map the long edge to this val

MAX_NEW_TOKENS=128 # {128, 200, 256}, 128 works well

model_id = "Qwen/Qwen3-VL-4B-Instruct"

all_cols = [
    'cigarettes',
    'cigars',
    'e-cigarettes',
    'gum',
    'heated_tobacco',
    'hookah',
    'lozenges',
    'patches',
    'pipe_tobacco',
    'smokeless_tobacco',
    'uncategorized'
]

prompts = {
    "caption": "Describe the item(s) in the image; look out for nicotine or tobacco related products. Return the descriptions (1 sentence), flavors, text, marketing strategy (1 sentence), shapes, and colors, in a JSON list. If an attribute does not exist, return an empty string for that attribute. Be factual. Template for one item: {'item': '', 'description':'', 'flavors':'', 'marketing':'', 'shape':'', 'color':'', 'text':''}",
    "sift": "Given the image, is there anything related to tobacco or nicotine? Consider tobacco or nicotine brands, products, people using products, advertisements, etc. Respond only with 'yes' or 'no'."
}

""" Example usage

python img_captions_vllm.py --input_csv "image_labels_positive_subset.csv" --output_dir "sift_results" --prompt "sift" --column_name "e-cigarettes"
"""
# Parse args
parser = argparse.ArgumentParser('VLLM Qwen batched captioning')
parser.add_argument('--input_csv', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--prompt', type=str, required=True, choices=prompts.keys())
parser.add_argument('--column_name', required=True, type=str, choices=all_cols)
args = parser.parse_args()

input_csv = args.input_csv
assert os.path.isfile(input_csv)
output_dir = args.output_dir
prompt = args.prompt
col = [args.column_name]
assert len(col) == 1


# Config
config = {
    #"data_labels_filepath": "/home/mserna/projects/tobacco-projects/smart_connect_health_neurips_2026/data/tobacco_1m_raw/tobacco_1m_2026.csv", # this was the path used for generating captions for all samples (many of which were noisy)
    "data_labels_filepath": input_csv,
    "results_path": f"/home/mserna/projects/tobacco-projects/smart_connect_health_neurips_2026/data/{output_dir}",
    "qwen_path": model_id,
}

# Create output dir
if not os.path.isdir(config["results_path"]):
    os.makedirs(config["results_path"])


# Read data, set output path
if prompt == "caption":
    df = pd.read_csv(config["data_labels_filepath"], index_col=0)
    df = df[df['product_type'].isin(col)].reset_index(drop=True)
    caption_prompt = prompts["caption"]
elif prompt == "sift":
    df = pd.read_csv(config["data_labels_filepath"])#, index_col=0)

    # With the data we used to generate captions, and the samples we want to clean up, we have object Ids, so only get id 0 and remove duplicate file paths
    df = df[df['object_id'] == 0]
    print(f"Samples: {len(df)}")

    # "/home/mserna/projects/tobacco-projects/smart_connect_health_neurips_2026"
    df.filepath = df.filepath.str.replace("positive_dataset", "/media/ttdat/Data2TB/manuel/tobacco/tobacco_1m_2026/positive_dataset")
    df = df[df['simple_product_type'].isin(col)].reset_index(drop=True)
    caption_prompt = prompts["sift"]
else:
    raise ValueError(f"Incorrect prompt: {prompt}")

print(f"[INFO] Caption prompt: {prompt} -> {caption_prompt}")

out_filepath = os.path.join(config["results_path"], f"res_qwen3vl_nodist_{col[0]}.json")
inter_filepath = os.path.join(config["results_path"], f"inter_qwen3vl_nodist_{col[0]}.json")


# Resume if inter file exists
if os.path.isfile(inter_filepath):
    progress_df = pd.read_json(inter_filepath)
    out_data:list = progress_df.to_dict(orient='records') # list of dicts, we resume now
    start = len(out_data) # start where we left off
    print(f"[INFO] Resuming inference for {col[0]}, already have {len(out_data)} captions")
else:
    start = 0
    out_data = []
    print(f"[INFO] Starting with new product type: {col[0]}.")


def write2json(write_path, data):
    with open(write_path, 'w') as file:
        file.write(json.dumps(data, indent=4))


# 1. Initialize the model
# For 48GiB VRAM, you can use FP16 or FP8.
# Use trust_remote_code=True for Qwen3 series support.
llm = LLM(
    model=model_id,
    trust_remote_code=True,
    dtype="float16",
    download_dir="/home/mserna/.cache/huggingface/hub",
    gpu_memory_utilization=0.8, # Reserve space for KV cache
    max_model_len=8196,          # Adjust based on expected output length
    limit_mm_per_prompt={"image": 1},
    enforce_eager=True
    #vllm_config={"torch_compile": False}
)


# 2. Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.2,
    max_tokens=256,
    stop_token_ids=[]
)


# 3. Prepare the batch of images
# Each entry is a list of messages following the OpenAI-like format
processor = AutoProcessor.from_pretrained(model_id)

def prepare_batch(paths):
    batch_inputs = []
    for path in paths:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": caption_prompt},
            ]
        }]
        # Format the prompt using the model's chat template
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Prepare multi-modal data
        # Note: vLLM expects a dictionary mapping modality name to data
        image_inputs, _ = process_vision_info(messages)

        batch_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs}
        })
    return batch_inputs


# Run inference
for ref_idx in tqdm(range(start, len(df), DELTA)):
    print(f"[{col[0]}] start={start}...ref_idx={ref_idx},DELTA={DELTA}...N={len(df)}")
    batch_filepaths = []
    uids = []

    for d in range(DELTA):
        if ref_idx+d >= len(df):
            print(f" > Reached the end of the samples set ({col[0]}).")
            continue # we have reached the end
        row = df.iloc[ref_idx+d]
        uids.append(int(row.uid))
        batch_filepaths.append(row.filepath)

    model_inputs = prepare_batch(batch_filepaths)
    outputs = llm.generate(model_inputs, sampling_params=sampling_params)

    for d in range(DELTA):
        if ref_idx+d >= len(df):
            continue
        out_text = outputs[d].outputs[0].text
        out_data.append({
            "filepath": batch_filepaths[d],
            "uid": uids[d], # important to trace back to other sample info
            "caption": out_text,
        })

    # Write intermediate results to file
    write2json(inter_filepath, out_data)

# Write final list to file
write2json(out_filepath, out_data)
print(f"[{col[0]}] Done. Processed {len(out_data)} samples ({len(df)} from dataframe).")
