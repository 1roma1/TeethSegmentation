import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import BinarySegmentationDataset
from train_bin_segmentation import train_bin_segmentation
from model import Unet


LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 15
NUM_WORKERS = 2
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

TRAIN_IMG_DIR = "dataset/imgs/train/"
TRAIN_MAKS_DIR = "dataset/maxillomandibular_masks/train/"
VAL_IMG_DIR = "dataset/imgs/val/"
VAL_MAKS_DIR = "dataset/maxillomandibular_masks/val/"
MODEL_PATH = "bin_segmentation_model.pt"


def main():
    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            interpolation=transforms.InterpolationMode.NEAREST),])

    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_ds = BinarySegmentationDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MAKS_DIR,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    val_ds = BinarySegmentationDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MAKS_DIR,
        transform=transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}")
        train_bin_segmentation(train_loader, val_loader, model,
                               optimizer, loss_fn, DEVICE, MODEL_PATH)


if __name__ == "__main__":
    main()
