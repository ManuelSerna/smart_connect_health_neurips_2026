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


WORLD_SIZE = 32 # number of GPUs to use, reduce to use less cards, NOTE: this has to agree with external world size (when we run from a bash script)

caption_prompt = "Describe the item(s) in the image; look out for nicotine or tobacco related products. Return the descriptions (1 sentence), flavors, text, marketing strategy (1 sentence), shapes, and colors, in a JSON list. If an attribute does not exist, return an empty string for that attribute. Be factual. Template for one item: {'item': '', 'description':'', 'flavors':'', 'marketing':'', 'shape':'', 'color':'', 'text':''}"

MAX_NEW_TOKENS=128 # {128, 200, 256}, 128 works well

DELTA = 64

MAX_IM_SIDE_LEN = 1000
MAX_ASSIGN_LEN = 854 # map the long edge to this val


def write2json(write_path, data):
    with open(write_path, 'w') as file:
        file.write(json.dumps(data, indent=4))


def get_qwen3vl_model(config:dict) -> tuple:
    """ Get a Qwen model

    NOTE:
    - 2B does not perform quite well...
    - 4B with float16 seems like a good middleground

    :param config: (dict) give path to Huggingface model, or local huggingface model
    :return: HF model
    """
    if config["task"] == "debug":
        model = AutoModelForImageTextToText.from_pretrained(
            config["qwen_path"],
            dtype=torch.bfloat16, # {"auto", torch.bfloat16}
            # attn_implementation="flash_attention_2",
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
            #max_len = imsize[1]
            new_size = (int(imsize[0]/imsize[1]*float(MAX_ASSIGN_LEN)), MAX_ASSIGN_LEN)
        else:
            #max_len = imsize[0]
            new_size = (MAX_ASSIGN_LEN, int(imsize[1]/imsize[0]*float(MAX_ASSIGN_LEN)))
        
        img = img.resize(new_size)
        
    #img = BytesIO(img)
    
    return img


def inference_qwen(config:dict):
    """
    Qwen3 VL model for image batch inference
    
    :param config: Description
    :param img_df: Description
    :type img_df: pd.DataFrame
    :param col: Name for the column called `product_type` in img_df
    :type col: str
    """
    # Read data
    df = pd.read_csv(config["data_labels_filepath"], index_col=0)

    print(f"[Qwen3-VL] Using Qwen3-VL model for VLM inference on n={len(df)}.")

    model, processor = get_qwen3vl_model(config)
    img_extract_prompt = caption_prompt
    print(f"[Qwen3-VL] Prompt:\n\t> `{img_extract_prompt}`")

    out_path = os.path.join(config["results_path"], f"res_qwen3vl_nodist.json")
    inter_filepath = os.path.join(config["results_path"], f"inter_qwen3vl_nodist.json")
    out_data = []
    bad = 0

    for row_idx in tqdm(range(len(df))):
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
        #inputs = inputs.cuda()#to("cuda")
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        current_data["description"] = output_text
        out_data.append(current_data)
        write2json(inter_filepath, out_data)

    print(f"\n[INFO] Files that could not be read: {bad}")
    write2json(out_data, out_data)


###############################################################
# Batched code
###############################################################
def prep_one_sample(row_idx, df):
    row = df.iloc[row_idx]
    in_filepath = row["filepath"]
    current_data = {
        "img_filepath": row["filepath"],
        "uid": int(row["uid"]),
        "tobacco_type": row["tobacco_type"],
        "product_type": row["product_type"],
        "product_name": row["product_name"],
    }

    try:
        img = prep_qwen_img(in_filepath)
    except Exception as e:
        print(f"An error occurred while opening the image {in_filepath}: {e}")
        # bad += 1
        #current_data["description"] = "-1"
        #out_data.append(current_data)
        return None

    # Messages containing multiple images and a text query
    current_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": caption_prompt},
            ],
        }
    ]

    return current_messages, current_data


def inference_qwen_batched(config: dict):
    """
    Qwen3 VL model for image batch inference

    :param config: Description
    """
    batched = True

    # Read data
    df = pd.read_csv(config["data_labels_filepath"], index_col=0)

    print(f"[Qwen3-VL] Using Qwen3-VL model for VLM inference on n={len(df)}.")

    model, processor = get_qwen3vl_model(config)
    if batched:
        processor.tokenizer.padding_size = "left"
    img_extract_prompt = caption_prompt
    print(f"[Qwen3-VL] Prompt:\n\t> `{img_extract_prompt}`")

    out_path = os.path.join(config["results_path"], f"res_qwen3vl_nodist.json")
    inter_filepath = os.path.join(config["results_path"], f"inter_qwen3vl_nodist.json")
    out_data = []
    bad = 0

    for row_idx in tqdm(range(0, len(df), DELTA)):
        messages = []
        data_batch = []
        for d in range(DELTA):
            current_messages, current_data = prep_one_sample(row_idx, df)
            messages.append(current_messages)
            data_batch.append(current_data)

        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        for d in range(DELTA):
            data_batch[d]["description"] = output_text[d]
            out_data.append(data_batch[d])

        write2json(inter_filepath, out_data)

    print(f"\n[INFO] Files that could not be read: {bad}")
    write2json(out_path, out_data)


###############################################################
# Distributed code
###############################################################
def inference_qwen_dist(config, dist_args):
    if dist_args.resume:
        # ...the below rank-to-file_idx mapping is hardcoded depending on the problem
        statuses = {
            "in_progress": [],
        }
        if len(statuses["in_progress"]) == 0:
            raise ValueError("If resuming, then manually fill in the ranks to continue!")

        _og_rank = int(os.environ["RANK"])
        rank = statuses["in_progress"][_og_rank]
        print(f"[DIST] Assigned rank: {_og_rank} --> Mapped rank (world size=32): {rank}")
    else:
        rank = int(os.environ["RANK"]) # NOTE: use for output names
    
    world_size = WORLD_SIZE # dist_args.world_size <--- 32 is number of original partitions, treat this as a constant
    
    if not args.debug:
        if not os.path.exists(config["results_path"]) and rank == 0:
            os.makedirs(config["results_path"])
    
    df = pd.read_csv(config["data_labels_filepath"], index_col=0)
    #product_types = ['cigarettes', 'heated_tobacco', 'e-cigarettes', 'smokeless_tobacco'] # NOTE: focus on subset at this point in time
    #df = df[df['product_type'].isin(product_types)].reset_index(drop=True)

    print(f"Product types: {df['product_type'].unique()}")

    # Dist setup
    # ...split df according to world_size
    N = len(df)
    n = N // world_size
    start_rank = n * rank # NOTE: may get overwritten if resuming...
    end_rank = start_rank + n # recall last idx is exclusive

    out_filepath = os.path.join(config["results_path"], f"res_qwen3vl_{rank}.json")
    inter_filepath = os.path.join(config["results_path"], f"inter_qwen3vl_{rank}.json")

    print(f"[DIST] ({rank}) results file: {out_filepath}; intermediate results: {inter_filepath}")

    # Resume
    if dist_args.resume:
        # Get previously-written records
        progress_df = pd.read_json(inter_filepath)
        out_data = progress_df.to_dict(orient='records')
    else:
        if not os.path.exists(config["results_path"]) and rank == 0:
            os.makedirs(config["results_path"])
        out_data:list[dict] = []
    
    # Get df gpu will work with
    if rank == world_size-1:
        df = df[start_rank:] # last slice
    else:
        df = df[start_rank:end_rank]
    
    print(f"[INFO] Local rank={rank}; World size={world_size}; Range: [{start_rank}, {end_rank})")

    start_idx = len(out_data)
    end_idx = len(df)

    if start_idx > 0:
        print(f"[INFO] Rank={rank}. Resuming from entry {start_idx}/{end_idx}")

    # Qwen setup
    model, processor = get_qwen3vl_model(config=config)
    img_extract_prompt = caption_prompt
    print(f"[Qwen3-VL] Prompt:\n\t> `{img_extract_prompt}`")

    # Iterate over all samples for Qwen inference
    bad_reads = 0

    #for row_idx in tqdm(range(start_idx, end_idx)):
    for row_idx in range(start_idx, end_idx):
        print(f"[rank={rank}][start={start_idx}...{row_idx}...{end_idx}]")
        row = df.iloc[row_idx]
        in_filepath = row["filepath"]
        current_data = {
            "uid": int(row["uid"]),
            "img_filepath": row["filepath"], 
            "tobacco_type": row["tobacco_type"], 
            "product_type": row["product_type"], 
            "product_name": row["product_name"], 
        }

        # Try to open image
        try:
            img = prep_qwen_img(in_filepath)
        except Exception as e:
            print(f"[WARNING] An error occurred while opening the image {in_filepath}: {e}")
            bad_reads += 1
            current_data["description"] = "-1"
            out_data.append(current_data)
            continue
        
        '''if dist_args.debug:
            current_data["description"] = "-1"
            out_data.append(current_data)
            write2json(inter_filepath, out_data)
            continue'''

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
        #inputs = inputs.cuda()#.to("cuda")
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        current_data["description"] = output_text
        out_data.append(current_data)

        # Save results so far
        write2json(inter_filepath, out_data)

    # Clean up and write results
    write2json(out_filepath, out_data)
    
    print(f"[INFO] Bad reads: {bad_reads}")
    print(f"[INFO] Wrote file: {out_filepath}")

    if dist_args.debug:
        print(f"[DEBUG] Process ({rank}/{world_size}) done!")
        return 
    
    print(f"[INFO] Process ({rank}/{world_size}) done!")
    #torch.distributed.barrier()


def setup_dist(args):
    args.gpu = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.dist_url = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    args.dist_backend = "nccl"
    args.resume = True ###### NOTE: hardcoding this in dist environment!!!

    if args.resume:
        print(f"[RESUME] Resuming progress...")

    if args.debug:
        print("[DEBUG] Not setting torch distributed settings in debug mode.")
        return
    
    print(f'| distributed init (rank {args.rank}/{args.world_size}): {args.dist_url}, gpu {args.gpu}')

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(days=365), 
    )
    torch.cuda.set_device(args.gpu)
    torch.distributed.barrier()


def main_dist(args):
    msize = "4B"

    if args.debug:
        print("[DEBUG] Setting script to DEBUG mode!")
        config = {
            "task": "debug",
            "data_labels_filepath": "/home/mserna/projects/tobacco-projects/smart_connect_health_neurips_2026/data/tobacco_1m_raw/tobacco_1m_2026.csv",
            "results_path": "/home/mserna/projects/tobacco-projects/smart_connect_health_neurips_2026/data/debug_results",
            "qwen_path": f"Qwen/Qwen3-VL-{msize}-Instruct", 
        }
        #inference_qwen(config)
        inference_qwen_batched(config)
    else:
        print("[INFO] Setting script to run inference in a distributed environment!")

        config = {
            "task": "hpc",
            "data_labels_filepath": "/scrfs/storage/mserna/home/Programming/tobacco_1m_2026/hpc_image_labels.csv",
            "results_path": "/scrfs/storage/mserna/home/Programming/tobacco_1m_2026/results/detailed_captions-qwen_vlm",
            "qwen_path": f"Qwen/Qwen3-VL-{msize}-Instruct",
        }
        setup_dist(args)
        inference_qwen_dist(config=config, dist_args=args)

    print("[INFO] Done!")


if __name__ == "__main__":
    # Start by setting cmd args, in a dist setting, these args will be populated, and debugging will export env variables, and not set the below flags
    parser = argparse.ArgumentParser('slurm training')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    
    main_dist(args)
