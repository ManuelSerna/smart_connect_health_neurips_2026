import os
import open_clip
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


pretrained_models_vig = {
    "pvig_b": "pvig_b_83.66.pth.tar", # "pvig_b_224_gelu"
    "pvig_m": "pvig_m_83.1.pth.tar", # "pvig_m_224_gelu",
    "pvig_s": "pvig_s_82.1.pth", # "pvig_s_224_gelu",
    "pvig_ti": "pvig_ti_78.5.pth.tar", # "pvig_ti_224_gelu",
    "vig_b": "vig_b_82.6.pth", # "vig_b_224_gelu"
    "vig_s": "vig_s_80.6.pth", # "vig_s_224_gelu"
    "vig_ti": "vig_ti_74.5.pth" #"vig_ti_224_gelu"
}

pretrained_models_open_clip = {
('RN50','openai'),
('RN101', 'openai'),

('convnext_base', 'laion400m_s13b_b51k'),
('convnext_base_w', 'laion2b_s13b_b82k'),

('coca_ViT-B-32', 'laion2b_s13b_b90k'),
('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'),

('ViT-B-16', 'laion2b_s34b_b88k'),
('ViT-B-16-SigLIP2-256', 'webli'),
}


def get_transforms(source, model_name):
    # Return transforms for all images
    if source == 'open-clip':
        if "SigLIP2" in model_name:
            resize_precrop = (278, 278)
            resize_dim = (256, 256)
            print(f"Using SigLIP2 dims: 278 -> 256")
        else:
            resize_precrop = (256, 256)
            resize_dim = (224, 224)
            print("Using regular dims: 256 -> 224")

        t_mean = (0.48145466, 0.4578275, 0.40821073)
        t_stdev = (0.26862954, 0.26130258, 0.27577711)
    else:
        resize_precrop = (256, 256)
        resize_dim = (224, 224)
        t_mean = (0.485, 0.456, 0.406)
        t_stdev = (0.229, 0.224, 0.225)

    # Set
    train_transform = transforms.Compose([
        transforms.Resize(resize_precrop),
        transforms.RandomCrop(resize_dim),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=t_mean, std=t_stdev),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=t_mean, std=t_stdev)
    ])

    return train_transform, val_test_transform


class TobaccoMultimodalDataset(Dataset):
    def __init__(self, df_path, transforms=None):
        self.data = pd.read_csv(df_path, index_col=0)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx].filepath).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image


def get_all_data_loader(df_path):
    all_dataset = TobaccoMultimodalDataset(df_path)
    dataloader = DataLoader(all_dataset, batch_size=1, shuffle=False, num_workers=0)

    return dataloader