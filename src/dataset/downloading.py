import os
import zipfile
import urllib.request
import urllib.error
import progressbar


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


def download_files(config):
    os.makedirs(config["path_to_raw_data"], exist_ok=True)

    for zip_file in config["zip_files"]:
        print(f"Downloading {zip_file} ...")
        try:
            urllib.request.urlretrieve(
                config["url"] + zip_file, zip_file, DownloadProgressBar()
            )
        except urllib.error.HTTPError as e:
            print(e)

        print(f"Unzip {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as f:
            f.extractall(config["path_to_raw_data"])
        os.remove(zip_file)
