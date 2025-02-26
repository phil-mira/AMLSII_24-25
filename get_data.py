import os
import shutil
import glob
import kaggle
import pandas as pd
import json
import zipfile
from PIL import Image
import io


def process_image(image_names, file_name, competition_name):
    """
    Function that downloads required images from kaggle.

    Args:
        image_names - required images to be downloaded
        file_name - local file path for images folder
        competition_name - competitoin path from kaggle

    """

    dataset = file_name.split(".")[0]
    image_folder_path = ("data/" + dataset + "_images/")
    # Download and process selected images
    for image in image_names:
        image_zip_path = image_folder_path + image + ".jpg.zip"
        kaggle.api.competition_download_file(
            competition_name, f"jpeg/{dataset}/{image}.jpg", path=image_folder_path)

        # Unzip the downloaded image
        if os.path.exists(image_zip_path):
            with zipfile.ZipFile(image_zip_path, "r") as z:
                z.extractall(image_folder_path)
            os.remove(image_zip_path)

            # Reduce resolution and compress
            image_path = image_folder_path + image + ".jpg"
            with Image.open(image_path) as img:
                # Resize to 12.5% of original size
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
                # Save with compression
                img.save(image_path, "JPEG", quality=60)


def get_files_process(competition_name, file_name, samples):
    """
    Args:
        competition_name - competitoin path from kaggle
        file_name - local file path for images folder
        samples - how many samples of dataset to include

    """

    csv_path = ("data/" + file_name)

    # Download only the required file
    kaggle.api.competition_download_file(
        competition_name, file_name, path="data/")

    # Extract the file if needed
    zip_file = "data/" + file_name + ".zip"
    if os.path.exists(zip_file):
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall("data/")
        os.remove(zip_file)

    df = pd.read_csv(csv_path, nrows=samples)
    image_names = df["image_name"].tolist()

    process_image(image_names, file_name, competition_name)


def run(train_samples, test_samples):

    os.makedirs("data/train_images", exist_ok=True)
    os.makedirs("data/test_images", exist_ok=True)

    # Define competition name
    competition_name = "siim-isic-melanoma-classification"
    train_file = "train.csv"
    test_file = "test.csv"

    get_files_process(competition_name, train_file, samples=train_samples)
    get_files_process(competition_name, test_file, samples=test_samples)
