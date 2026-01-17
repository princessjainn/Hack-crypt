import os
import torch
from torch.utils.data import Dataset
import cv2
from backend.preprocessing.image import transform, transform_train

class ImageDeepfakeDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.samples = []
        self.root_dir = os.path.abspath(root_dir)
        split_lower = split.lower()
        split_map = {
            "train": "Train",
            "training": "Train",
            "val": "Validation",
            "validation": "Validation",
            "test": "Test",
        }
        if split_lower not in split_map:
            raise ValueError(f"Invalid split: {split}. Use train/validation/test")

        split_dir = os.path.join(self.root_dir, split_map[split_lower])
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        # Use augmentations only for training
        self.transform = transform_train if split_lower == "train" else transform

        for label, cls in enumerate(["real", "fake"]):
            cls_path = os.path.join(split_dir, cls)
            if not os.path.exists(cls_path):
                raise FileNotFoundError(f"Missing folder: {cls_path}")

            for file in os.listdir(cls_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(cls_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Invalid image path: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(img)
        return image, torch.tensor(label, dtype=torch.long)
