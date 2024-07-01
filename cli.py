import click

from src.utils import load_configuration
from src.dataset.downloading import download_files
from src.dataset.split import create_segmentation_dataset
from src.training.train import train_model


@click.command()
def download():
    config = load_configuration("conf/config.yaml")
    download_files(config)


@click.command()
def split():
    config = load_configuration("conf/config.yaml")
    create_segmentation_dataset(config)


@click.command()
def train():
    config = load_configuration("conf/config.yaml")
    train_model(config)


@click.group()
def cli():
    pass


cli.add_command(download)
cli.add_command(split)
cli.add_command(train)


if __name__ == "__main__":
    cli()
