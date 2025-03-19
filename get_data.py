import os
import shutil
import glob
import pandas as pd
import json
import zipfile
from PIL import Image
import io



def verify_image_sizes(image_folder_path):
    """
    Checks if all images in the folder have been properly resized to 256x256.
    
    Args:
        image_folder_path - path to folder containing images
    
    Returns:
        bool: True if all images are 256x256, False otherwise
    """
    all_correct_size = True
    image_files = glob.glob(os.path.join(image_folder_path, "*.jpg"))
    
    for image_path in image_files:
        with Image.open(image_path) as img:
            width, height = img.size
            if width != 256 or height != 256:
                print(f"Warning: {os.path.basename(image_path)} is {width}x{height}, not 256x256")


    if all_correct_size:
        print(f"All images in {image_folder_path} are correctly sized at 256x256")
    else:
        print(f"Some images in {image_folder_path} are not properly sized")
    
    return all_correct_size


def get_files_process(csv_name, samples):
    """
    Args:
        competition_name - competitoin path from kaggle
        file_name - local file path for images folder
        samples - how many samples of dataset to include

    """

    # Extract the file if needed
    csv_file = "./data/" + csv_name 
    df = pd.read_csv(csv_file, nrows=samples)
    image_names = df["image_name"].tolist()

    dataset = csv_name.split(".")[0]
    extract_path = f"./data/{dataset}_images"
  

    for image in image_names:
        image_path = f"./data/siim-isic-melanoma-classification/jpeg/{dataset}/{image}.jpg"
        with Image.open(image_path) as img:
            # Resize to 12.5% of original size
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            # Save with compression
            output_path = os.path.join(extract_path, f"{image}.jpg")
            img.save(output_path, "JPEG", quality=60)

    

def run(train_samples, test_samples):

    os.makedirs("data/train_images", exist_ok=True)
    os.makedirs("data/test_images", exist_ok=True)

    # Define competition name
   
    train_file = "train.csv"
    test_file = "test.csv"

    get_files_process(train_file, samples=train_samples)
    get_files_process(test_file, samples=test_samples)


    verify_image_sizes("data/train_images")
    verify_image_sizes("data/test_images")
