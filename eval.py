import argparse
from tqdm import tqdm

from data.dataset import *


eval_task_choices = {
    "precompute_img_features": "Precompute image features with all models.",
}


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=list(eval_task_choices.keys()))

    args = parser.parse_args()

    return args


def precompute_img_features():
    print("[INFO] Pre-computing image features...")

    df_path = "data/simple_image_labels.csv"
    data_loader = get_all_data_loader(df_path)

    img_features = self.model.encode_image(x)

    for batch in tqdm(data_loader):
        pass

    import pdb;pdb.set_trace()

    print("[INFO] Done.")


def main():
    args = get_eval_args()

    if args.task == "precompute_img_features":
        precompute_img_features()
    else:
        raise NotImplementedError(f"Tasks: {eval_task_choices}")


if __name__ == '__main__':
    main()