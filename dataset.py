# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils import (
    rgb_to_lab,
    split_lab,
    generate_hints,
    ab_to_bin,
    to_tensor
)


class ColorizationDataset(Dataset):
    def __init__(
        self,
        root="./cat-and-dog",
        train=True,
        hint_ratios=(0.01, 0.03, 0.05),
        max_samples=5000,
        image_size=256,
        cache_dir="cache_lab_bins"
    ):
        self.hint_ratios = hint_ratios
        self.image_size = image_size
        self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)

        split = "train" if train else "test"
        self.image_paths = []

        for cls in ["cats", "dogs"]:
            cls_dir = os.path.join(root, split, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Directory not found: {cls_dir}")

            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.image_paths.append(os.path.join(cls_dir, fname))

        np.random.shuffle(self.image_paths)

        if max_samples is not None:
            self.image_paths = self.image_paths[:max_samples]

        print(f"[Dataset] Loaded {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.basename(img_path)
        name = os.path.splitext(base)[0]
        cache_path = os.path.join(self.cache_dir, f"{name}_{self.image_size}.npz")

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            L = data["L"]
            ab = data["ab"]
            bin_idx = data["bin"]
        else:
            img = Image.open(img_path).convert("RGB")
            if self.image_size is not None:
                img = img.resize(
                    (self.image_size, self.image_size),
                    Image.BILINEAR
                )

            rgb = np.array(img, dtype=np.uint8)

            lab = rgb_to_lab(rgb)
            L, ab = split_lab(lab)
            ab = np.clip(ab, -110, 110)

            bin_idx = ab_to_bin(ab)[0]
            bin_idx = np.clip(bin_idx, 0, 312)

            np.savez(
                cache_path,
                L=L,
                ab=ab,
                bin=bin_idx
            )

        _, H, W = ab.shape
        hint_ratio = np.random.choice(self.hint_ratios)
        num_hints = int(hint_ratio * H * W)

        ab_hint, mask = generate_hints(ab, num_hints)

        return (
            to_tensor(L),
            to_tensor(ab_hint),
            to_tensor(mask),
            torch.from_numpy(bin_idx).long(),
            to_tensor(ab)
        )
