import argparse
import datetime
import json
import os
import pandas as pd
from PIL import Image
import time
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor # get transformers == 4.57.0


#caption_prompt = "Describe the top-5 possible tobacco or nicotine related products, or item if no tobacco product is present. Return the descriptions, flavors, text, shapes, and colors in a JSON list."
caption_prompt = "Describe the item(s) in the image; look out for nicotine or tobacco related products. Return the descriptions (1 sentence), flavors, text, marketing strategy (1 sentence), shapes, and colors, in a JSON list. If an attribute does not exist, return an empty string for that attribute. Be factual. Template for one item: {'item': '', 'description':'', 'flavors':'', 'marketing':'', 'shape':'', 'color':'', 'text':''}"


MAX_IM_SIDE_LEN = 1000
MAX_ASSIGN_LEN = 854 # map the long edge to this val


def write2json(write_path, data):
    with open(write_path, 'w') as file:
        file.write(json.dumps(data, indent=4))


def get_qwen3vl_model(config:dict) -> tuple:
    # default: Load the model on the available device(s)
    # Can choose from: {2B, 4B, 8B, 30B, 235B}
    #msize = "8B" # NOTE: to process 60k images, running inference will take several hundred hrs...unless we massively parallelize ops. Maybe we can use this for single, not batched, inference.
    # 4B seems just right
    #msize = "2B" # quite underperforming...
    if config["task"] == "debug":
        model = AutoModelForImageTextToText.from_pretrained(
            config["qwen_path"],
            #dtype="auto",
            dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2",
            attn_implementation="eager",
            device_map="auto"
        )
    elif config["task"] == "hpc":
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        model = AutoModelForImageTextToText.from_pretrained(
            config["qwen_path"],
            dtype=torch.bfloat16,
            #attn_implementation="flash_attention_2",
            attn_implementation="eager",
            device_map="auto"
        )
    else:
        raise ValueError(f"Field `task` in config is incorrect. Got: {config['task']}")

    processor = AutoProcessor.from_pretrained(config["qwen_path"])

    return model, processor


def prep_qwen_img(path):
    img = Image.open(path)

    # Check if we should resize very large images
    imsize = img.size

    if max(imsize) > MAX_IM_SIDE_LEN:
        if imsize[1] > imsize[0]:
            # max_len = imsize[1]
            new_size = (int(imsize[0] / imsize[1] * float(MAX_ASSIGN_LEN)), MAX_ASSIGN_LEN)
        else:
            # max_len = imsize[0]
            new_size = (MAX_ASSIGN_LEN, int(imsize[1] / imsize[0] * float(MAX_ASSIGN_LEN)))

        img = img.resize(new_size)

    # img = BytesIO(img)

    return img


def check_dataset():
    # Config
    old_root = "/scrfs/storage/mserna/home/Programming/tobacco_1m_2026/"
    new_root = ""
    df_path = "hpc_image_labels.csv"
    df = pd.read_csv(df_path, index_col=0)
    df.filepath = df.filepath.str.replace(old_root, new_root)

    # ...check that files exist
    # for j in tqdm(range(len(df))):
    #     row = df.iloc[j]
    #     assert os.path.isfile(row.filepath)

    rank = 0
    col = "all"
    config = {
        "task": "debug",
        "data_labels_filepath": df_path,
        "results_path": "debug_results",
        "qwen_path": f"Qwen/Qwen3-VL-4B-Instruct",
    }

    # Setup
    model, processor = get_qwen3vl_model(config)
    img_extract_prompt = caption_prompt

    out_path = os.path.join(config["results_path"], f"qwen3vl_{col}.json")
    inter_filepath = os.path.join(config["results_path"], f"inter_qwen3vl_{rank}.json")
    out_data = []
    bad = 0

    # Inference
    for row_idx in tqdm(range(0, len(df), 12000)):
        row = df.iloc[row_idx]
        in_filepath = row["filepath"]
        current_data = {
            "img_filepath": row["filepath"],
            "tobacco_type": row["tobacco_type"],
            "product_type": row["product_type"],
            "product_name": row["product_name"],
        }

        try:
            img = prep_qwen_img(in_filepath)
        except Exception as e:
            print(f"An error occurred while opening the image {in_filepath}: {e}")
            bad += 1
            current_data["description"] = "-1"
            out_data.append(current_data)
            continue

        # Messages containing multiple images and a text query
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": img_extract_prompt},
                ],
            }
        ]

        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        # inputs = inputs.cuda()#to("cuda")
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)

        current_data["description"] = output_text
        out_data.append(current_data)

        write2json(inter_filepath, out_data)

        # if row_idx >= 16:
        #    print("Getting out loop")
        #    break

    print(f"\n[INFO] Files that could not be read: {bad}")

    with open(out_path, 'w') as file:
        file.write(json.dumps(out_data, indent=4))

    import pdb;pdb.set_trace()


if __name__ == "__main__":
    check_dataset()