from pathlib import Path

import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.functional.classification import binary_jaccard_index
from torch.utils.tensorboard import SummaryWriter

from dataset import TeethSegmentationDataset
from unet import Unet


def train_bin_segmentation(
    train_loader, val_loader, model, optimizer, loss_fn, device, path
):
    loop = tqdm(train_loader)

    train_loss, train_iou, train_acc = 0, 0, 0
    val_loss, val_iou, val_acc = 0, 0, 0

    train_num_correct = 0
    train_num_pixels = 0
    model.train()
    for _, (img, mask) in enumerate(loop):
        img = img.to(device)
        mask = mask.float().unsqueeze(1).to(device)

        predictions = model(img)
        loss = loss_fn(predictions, mask)
        iou = binary_jaccard_index(predictions, mask)

        preds = torch.sigmoid(predictions)
        preds = (preds > 0.5).float()
        train_num_correct += (preds == mask).sum()
        train_num_pixels += torch.numel(preds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(train_loss=loss.item())
        train_loss += loss.item()
        train_iou += iou.item()
    train_acc = train_num_correct / train_num_pixels

    val_num_correct = 0
    val_num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)
            preds = model(x)
            loss = loss_fn(preds, y)
            iou = binary_jaccard_index(preds, y)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()
            val_num_correct += (preds == y).sum()
            val_num_pixels += torch.numel(preds)

            val_loss += loss.item()
            val_iou += iou.item()
        val_acc = val_num_correct / val_num_pixels

    train_loss /= len(train_loader)
    train_iou /= len(train_loader)
    val_loss /= len(val_loader)
    val_iou /= len(val_loader)

    print(
        f"train_loss: {train_loss} train_iou: {train_iou}"
        f"train_acc: {train_acc}"
    )
    print(f"val_loss: {val_loss} val_iou: {val_iou} val_acc: {val_acc}")
    torch.save(model.state_dict(), path)

    return train_loss, train_iou, train_acc, val_loss, val_iou, val_acc


def train_model(config):
    data_path = Path(config["path_to_processed_data"])
    transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_height"], config["image_width"]),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
        ]
    )

    model = Unet(in_channels=1, out_channels=1).to(config["device"])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    train_ds = TeethSegmentationDataset(
        image_dir=data_path / "imgs/train/",
        mask_dir=data_path / "teeth_masks/train/",
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
    )

    val_ds = TeethSegmentationDataset(
        image_dir=data_path / "imgs/val/",
        mask_dir=data_path / "teeth_masks/val/",
        transform=transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
    )

    writer = SummaryWriter()

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch + 1}")
        train_loss, train_iou, train_acc, val_loss, val_iou, val_acc = (
            train_bin_segmentation(
                train_loader,
                val_loader,
                model,
                optimizer,
                loss_fn,
                config["device"],
                config["model_path"],
            )
        )

        writer.add_scalars(
            "Loss", {"train_loss": train_loss, "val_loss": val_loss}, epoch + 1
        )
        writer.add_scalars(
            "Intersection over union",
            {"train_iou": train_iou, "val_iou": val_iou},
            epoch + 1,
        )
        writer.add_scalars(
            "Accuracy", {"train_acc": train_acc, "val_acc": val_acc}, epoch + 1
        )
    writer.close()
