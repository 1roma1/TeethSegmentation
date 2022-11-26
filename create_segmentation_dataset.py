import os
import random
import shutil


RADIOGRAPHS_DIR = "data/Radiographs/"
TEETH_BIN_MASKS_DIR = "data/Segmentation/teeth_mask/"

RADIOGRAPHS_TRAIN_DIR = "dataset/imgs/train/"
RADIOGRAPHS_VAL_DIR = "dataset/imgs/val/"

TEETH_BIN_MASKS_TRAIN_DIR = "dataset/teeth_masks/train/"
TEETH_BIN_MASKS_VAL_DIR = "dataset/teeth_masks/val/"


def train_val_split(images):
    random.seed(42)
    random.shuffle(images)
    train_data, val_data = images[:900], images[900:]
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


if __name__ == "__main__":
    images = sorted(os.listdir(RADIOGRAPHS_DIR))

    train_data, val_data = train_val_split(images)

    copy_images(train_data, RADIOGRAPHS_DIR, TEETH_BIN_MASKS_DIR,
                RADIOGRAPHS_TRAIN_DIR, TEETH_BIN_MASKS_TRAIN_DIR)
    copy_images(val_data, RADIOGRAPHS_DIR, TEETH_BIN_MASKS_DIR,
                RADIOGRAPHS_VAL_DIR, TEETH_BIN_MASKS_VAL_DIR)
