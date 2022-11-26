import torch
from tqdm import tqdm


def train_bin_segmentation(train_loader, val_loader, model,
                           optimizer, loss_fn, device, path):
    loop = tqdm(train_loader)

    model.train()
    for _, (img, mask) in enumerate(loop):
        img = img.to(device)
        mask = mask.float().unsqueeze(1).to(device)

        predictions = model(img)
        loss = loss_fn(predictions, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(
        f"Got {num_correct}/{num_pixels} "
        f"with acc {num_correct/num_pixels*100:.2f}"
    )
    torch.save(model.state_dict(), path)
