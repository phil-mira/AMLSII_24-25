import os
import get_data
import json


import get_data
from preprocess import preprocess
from synthetic_data_generate import advanced_augmentation
from data_utils import create_dataloaders
from train_validate import train

from models import MixedInputModel

import kaggle
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np


def main():

    # Navigate to .kaggle directory and load credentials
    kaggle_path = os.path.join(os.path.expanduser('~'), '.kaggle')
    if not os.path.exists(kaggle_path):
        os.makedirs(kaggle_path)

    # Load kaggle.json file
    with open(os.path.join(kaggle_path, 'kaggle.json')) as f:
        api_token = json.load(f)

    # Check if data already exists
    train_folder = "data/train_images"
    test_folder = "data/test_images"
    
    # Create directories if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Count existing files in train and test folders
    train_files = len([f for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))])
    test_files = len([f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))])
    
    # Only download data if files are missing
    train_samples = 50
    test_samples = 10
    
    if train_files < train_samples or test_files < test_samples:
        print("Downloading data...")
        get_data.run(train_samples, test_samples)
    else:
        print("Data already exists, skipping download")

    train_data, val_data, test_data = preprocess()

    num_augmentations = 1
    train_dataset_path = "data/train_images"
    synth_dir = "data/synthetic_images"
    
    # Create synthetic images directory if it doesn't exist
    os.makedirs(synth_dir, exist_ok=True)
    
    # Count existing synthetic images
    synthetic_files = len([f for f in os.listdir(synth_dir) if os.path.isfile(os.path.join(synth_dir, f))])
    expected_synthetic_files = len(train_data) * num_augmentations
    
    if synthetic_files < expected_synthetic_files:
        print(f"Generating {expected_synthetic_files} synthetic images...")
        synth_data = advanced_augmentation(train_dataset_path, synth_dir, train_data, augmentations_per_image=num_augmentations)
    else:
        print("Synthetic images already exist, skipping generation")


    # Create additional dataset with all the synthetic images 
    use_synth = True
    if use_synth == True:
        train_data = synth_data
        # Move synthetic images to train images folder
        print("Moving synthetic images to train folder...")
        synth_image_files = [f for f in os.listdir(synth_dir) if os.path.isfile(os.path.join(synth_dir, f))]
        for file in synth_image_files:
            src_path = os.path.join(synth_dir, file)
            dst_path = os.path.join(train_dataset_path, file)
            if not os.path.exists(dst_path):  # Avoid overwriting existing files
                os.rename(src_path, dst_path)  # Move file (rename operation)
        print(f"Moved {len(synth_image_files)} synthetic images to training directory")



    experiment = True
    if experiment:
        print("Filtering datasets to include only rows with existing images...")
        # Get list of available images in train and test directories
        train_images = {f.split('.')[0] for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))}
        test_images = {f.split('.')[0] for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))}
        all_available_images = train_images.union(test_images)

        # Filter dataframes to only include rows with available images
        train_data = train_data[train_data['image_name'].str.split('.').str[0].isin(all_available_images)].reset_index(drop=True)
        val_data = val_data[val_data['image_name'].str.split('.').str[0].isin(all_available_images)].reset_index(drop=True)
        test_data = test_data[test_data['image_name'].str.split('.').str[0].isin(all_available_images)].reset_index(drop=True)

        print(f"After filtering: {len(train_data)} training samples, {len(val_data)} validation samples, {len(test_data)} test samples")



    model = MixedInputModel()

    # Define hyperparameters
    batch_size = 4
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = 'saved_models'

    tabular_features = train_data.columns[2:-1].to_list()

    train_loader, val_loader = create_dataloaders(train_data, val_data, train_dataset_path, 
                       img_id_column='image_name',
                       target_column='target',
                       tabular_columns=tabular_features,
                       batch_size=batch_size
                       )
    

    
    """    

    # Create directory for saving models if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on {device} with {len(train_data)} training samples, {len(val_data)} validation samples")


    train(model, train_loader, val_loader, criterion, optimizer, 
                     device, num_epochs, model_dir, checkpoint_freq=1, save_best_only=True, 
                     early_stopping_patience=None, start_epoch=0)
    """

if __name__ == "__main__":

    main()
