import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from dataset import TeethSegmentationDataset
from train_bin_segmentation import train_bin_segmentation
from model import Unet


LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 15
NUM_WORKERS = 2
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

TRAIN_IMG_DIR = "dataset/imgs/train/"
TRAIN_MAKS_DIR = "dataset/teeth_masks/train/"
VAL_IMG_DIR = "dataset/imgs/val/"
VAL_MAKS_DIR = "dataset/teeth_masks/val/"
MODEL_PATH = "bin_segmentation_model.pt"


def main():
    transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_HEIGHT, IMAGE_WIDTH),
            interpolation=transforms.InterpolationMode.NEAREST),])

    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_ds = TeethSegmentationDataset(
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

    val_ds = TeethSegmentationDataset(
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

    writer = SummaryWriter()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}")
        train_loss, train_iou, train_acc, val_loss, val_iou, val_acc = \
            train_bin_segmentation(train_loader, val_loader, model,
                                   optimizer, loss_fn, DEVICE, MODEL_PATH)

        writer.add_scalars('Loss',
                           {'train_loss': train_loss, 'val_loss': val_loss},
                           epoch+1)
        writer.add_scalars('Intersection over union',
                           {'train_iou': train_iou, 'val_iou': val_iou},
                           epoch+1)
        writer.add_scalars('Accuracy',
                           {'train_acc': train_acc, 'val_acc': val_acc},
                           epoch+1)
    writer.close()


if __name__ == "__main__":
    main()
