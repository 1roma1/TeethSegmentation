import torch
from tqdm import tqdm
from torchmetrics.functional.classification import binary_jaccard_index


def train_bin_segmentation(train_loader, val_loader, model,
                           optimizer, loss_fn, device, path):
    loop = tqdm(train_loader)

    train_loss = 0
    train_iou = 0
    train_acc = 0
    val_loss = 0
    val_iou = 0
    val_acc = 0

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
    train_acc = train_num_correct/train_num_pixels

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
        val_acc = val_num_correct/val_num_pixels

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
