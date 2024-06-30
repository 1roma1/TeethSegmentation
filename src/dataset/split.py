import os
import random
import shutil

from pathlib import Path


def train_test_split(images, seed):
    random.seed(seed)
    random.shuffle(images)
    train_data, val_data = images[:800], images[800:]
    return train_data, val_data


def copy_images(images, img_src, mask_src, img_dst, mask_dst):
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    for img in images:
        img_src_path = os.path.join(img_src, img)
        mask_src_path = os.path.join(mask_src, img.lower())

        img_dst_path = os.path.join(img_dst, img)
        mask_dst_path = os.path.join(mask_dst, img.lower())

        shutil.copy(img_src_path, img_dst_path)
        shutil.copy(mask_src_path, mask_dst_path)


def create_segmentation_dataset(config):
    raw_dir = config["path_to_raw_data"]
    processed_dir = config["path_to_processed_data"]

    radiographs_dir = Path(raw_dir) / "Radiographs"
    bin_masks_dir = Path(raw_dir) / "Segmentation/teeth_mask/"

    radiographs_train_dir = Path(processed_dir) / "imgs/train/"
    radiographs_val_dir = Path(processed_dir) / "imgs/val/"

    bin_masks_train_dir = Path(processed_dir) / "teeth_masks/train/"
    bin_masks_val_dir = Path(processed_dir) / "teeth_masks/val/"

    images = sorted(os.listdir(radiographs_dir))

    train_data, test_data = train_test_split(images, config["seed"])

    copy_images(
        train_data,
        radiographs_dir,
        bin_masks_dir,
        radiographs_train_dir,
        bin_masks_train_dir,
    )
    copy_images(
        test_data,
        radiographs_dir,
        bin_masks_dir,
        radiographs_val_dir,
        bin_masks_val_dir,
    )
