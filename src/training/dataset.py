import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class TeethSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5)
        image = clahe.apply(image)

        image = Image.fromarray(image)
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = np.array(mask, dtype=np.float32)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask[mask == 255.0] = 1.0

        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(mask)

        return image, mask
