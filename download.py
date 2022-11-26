import os
import zipfile
import urllib.request
import urllib.error
import progressbar


URL = "http://tdd.ece.tufts.edu/Tufts_Dental_Database/"
ZIP_FILES = ("Radiographs.zip", "Segmentation.zip")
DATA_DIR = "data/"


class DownloadProgressBar:
    """Progress bar for track file download progress."""

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    for zip_file in ZIP_FILES:
        print(f"Downloading {zip_file} ...")
        try:
            urllib.request.urlretrieve(
                URL+zip_file,
                zip_file,
                DownloadProgressBar()
            )
        except urllib.error.HTTPError as e:
            print(e)

        print(f"Unzip {zip_file}")
        with zipfile.ZipFile(zip_file, 'r') as f:
            f.extractall(DATA_DIR)
        os.remove(zip_file)
