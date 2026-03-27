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


image_feature_extraction_prompts = {
    #"basic": "Find all possible tobacco or nicotine related products, and return their names, shape, color, and any text as a JSON list.",
    #"basic2": "Find all possible tobacco or nicotine related products. Return their description, shape, color, and a confidence score between 0.0 and 100.0 as a JSON list.",
    "basic3": "Find the top-5 possible tobacco or nicotine related products. Return their description, shape, color, and a confidence score between 0.0 and 100.0 as a JSON list."
}

MAX_IM_SIDE_LEN = 1000
MAX_ASSIGN_LEN = 854 # map the long edge to this val


def get_qwen3vl_model(config:dict) -> tuple:
    # default: Load the model on the available device(s)
    # Can choose from: {2B, 4B, 8B, 30B, 235B}
    #msize = "8B" # NOTE: to process 60k images, running inference will take several hundred hrs...unless we massively parallelize ops. Maybe we can use this for single, not batched, inference.
    # 4B seems just right
    #msize = "2B" # quite underperforming...
    if config["task"] == "debug":
        model = AutoModelForImageTextToText.from_pretrained(
            config["qwen_path"], 
            dtype="auto", 
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


def prep_qwen_img(path, logger=None):
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


def get_img_ft_extract_prompt(prompt_key:str=None):
    assert prompt_key is not None, f"Choose from one of key-prompt pairs:\n{image_feature_extraction_prompts}"

    return image_feature_extraction_prompts[prompt_key]


def qwen_inference(config:dict, logger, img_df:pd.DataFrame, col:str=None):
    """
    Qwen3 VL model for image batch inference
    
    :param config: Description
    :type config: dict
    :param logger: Description
    :param img_df: Description
    :type img_df: pd.DataFrame
    :param col: Name for the column called `product_type` in img_df
    :type col: str
    """
    assert col is not None
    if col == "all":
        df = img_df
    else:
        df = img_df[img_df["product_type"] == col]

    logger.info(f"[Qwen3-VL] Using Qwen3-VL model for VLM inference on n={len(df)} images (column={col}).")

    model, processor = get_qwen3vl_model()
    img_extract_prompt = get_img_ft_extract_prompt(prompt_key="basic3")
    logger.info(f"[Qwen3-VL] Prompt:\n\t> `{img_extract_prompt}`")

    out_path = os.path.join(config["experiment_write_path"], f"qwen3vl_{col}.json")
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
            logger.info(f"An error occurred while opening the image {in_filepath}: {e}")
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
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        #print(output_text)

        current_data["description"] = output_text
        out_data.append(current_data)

        #if row_idx >= 16:
        #    print("Getting out loop")
        #    break
    
    logger.info(f"\n[INFO] Files that could not be read: {bad}")
    
    with open(out_path, 'w') as file:
        file.write(json.dumps(out_data, indent=4))


def evaluate_image_description(config, logger, img_df):
    """
    Run experiment for Qwen3 VL inference

    Links:
    https://github.com/QwenLM/Qwen3-VL
    https://huggingface.co/collections/Qwen/qwen3-vl
    
    :param config: Description
    :type config: dict
    :param logger: Description
    :param img_df: Description
    :type img_df: pd.DataFrame
    """
    logger.info(f"[EVAL image description analysis] Starting experiment.")
    _start = time.time()

    # qwen_inference(config, logger, img_df, "all") # very slow...

    # Separate files according to `product_type` column, 
    logger.info(f'Case ...{config["experiment_name"]}')
    logger.info(f"WARNING! Running this on all images in `img_df` will take a very long time! For this component of knowledge construction, run in a distributed computing environment.")

    qwen_inference(config=config, logger=logger, img_df=img_df, col="all")
    
    _end = time.time()
    logger.info(f"[EVAL image description analysis] Finished experiment. Elapsed time: {_end-_start:.3f} s")


###############################################################
# Distributed code
###############################################################
def write2json(write_path, data):
    with open(write_path, 'w') as file:
        file.write(json.dumps(data, indent=4))


def inference_qwen_dist(config, dist_args):
    if dist_args.resume:
        # ...the below rank-to-file_idx mapping is hardcoded depending on the problem
        statuses = {
            "in_progress": [9,2,0,1,10,3,11,8], 
        }
        _og_rank = int(os.environ["RANK"])
        rank = statuses["in_progress"][_og_rank]
        print(f"[DIST] Assigned rank: {_og_rank} --> Mapped rank (world size=32): {rank}")
    else:
        rank = int(os.environ["RANK"]) # NOTE: use for output names
    
    world_size = 32 # dist_args.world_size <--- 32 is number of original partitions, treat this as a constant
    
    if not args.debug:
        if not os.path.exists(config["results_path"]) and rank == 0:
            os.makedirs(config["results_path"])
    
    df = pd.read_csv(config["data_labels_filepath"], index_col=0)
    product_types = ['cigarettes', 'heated_tobacco', 'e-cigarettes', 'smokeless_tobacco'] # NOTE: focus on subset at this point in time
    df = df[df['product_type'].isin(product_types)].reset_index(drop=True)

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
    img_extract_prompt = get_img_ft_extract_prompt(prompt_key="basic3")
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
        generated_ids = model.generate(**inputs, max_new_tokens=128)
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
    setup_dist(args)
    msize = "4B"

    if args.debug:
        print("[DEBUG] Setting script to DEBUG mode!")
        config = {
            "task": "debug",
            "data_labels_filepath": "/home/mserna/projects/tobacco-projects/smart_connect_health_rag/data/tobacco_1m_2025/image_labels.csv",
            "results_path": "/home/mserna/projects/tobacco-projects/smart_connect_health_rag/image_modules/qwen/debug_results",
            "qwen_path": f"Qwen/Qwen3-VL-{msize}-Instruct", 
        }
    else:
        config = {
            "task": "hpc",
            "data_labels_filepath": "/scrfs/storage/mserna/home/Programming/tobacco_1m_2026/hpc_image_labels.csv",
            "results_path": "/scrfs/storage/mserna/home/Programming/tobacco_1m_2026/results/entity_extract-qwen_vlm",
            "qwen_path": f"Qwen/Qwen3-VL-{msize}-Instruct",
        }

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
