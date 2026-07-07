"""Data loading for surface-defect classification.

Expects an ImageFolder layout, e.g. the public "Surface Crack Detection"
dataset (Ozgenel) with two classes::

    data_dir/
        Positive/   # cracked surfaces
        Negative/   # intact surfaces

Any two-class (or multi-class) folder dataset works the same way.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(data_dir, img_size=224, batch_size=64, val_split=0.2,
                  num_workers=2, seed=42):
    """Split one ImageFolder into train/val loaders with separate transforms."""
    base = datasets.ImageFolder(data_dir)
    n = len(base)
    n_val = int(n * val_split)
    n_train = n - n_val

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_full = datasets.ImageFolder(data_dir, transform=build_transforms(img_size, True))
    val_full = datasets.ImageFolder(data_dir, transform=build_transforms(img_size, False))

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, base.classes
