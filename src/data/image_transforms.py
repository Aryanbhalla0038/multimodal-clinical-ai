"""Image loading and augmentation for chest X-rays (CheXpert / MIMIC-CXR)."""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ImageNet stats — ViT/timm pretrained models expect these
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image_rgb(path: str | Path) -> Image.Image:
    """Loads a PNG/JPG X-ray as a 3-channel PIL image."""
    return Image.open(str(path)).convert("RGB")


def build_train_transform(image_size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Reverses ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(t.device)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(t.device)
    return (t * std + mean).clamp(0, 1)
