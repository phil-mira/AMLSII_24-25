import os
import get_data
import json


import get_data
from preprocess import preprocess, one_hot
from synthetic_data_generate import advanced_augmentation
from data_utils import create_dataloaders, TestDataset
from train_validate import train

from models import BaseModel

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision import transforms
from PIL import Image


def main():
    """
    Main function to handle data preparation, augmentation, and model training.
    This function performs the following tasks:
    1. Ensures the Kaggle API credentials are available and loads them.
    2. Checks if the required training and testing data exists locally.
       - If not, downloads the data using the `get_data.run` function.
    3. Preprocesses the data into training, validation, and testing sets.
    4. Generates synthetic images using advanced augmentation techniques.
       - Ensures synthetic images are created only if they do not already exist.
       - Moves the generated synthetic images to the training directory.
    5. Converts the datasets into one-hot encoded format.
    6. Trains a base model using the prepared datasets.
    Note:
    - The function assumes the presence of helper functions such as `get_data.run`, 
      `preprocess`, `synth_data`, `advanced_augmentation`, `one_hot`, and `base_model`.
    - Directory paths and dataset sizes are hardcoded but can be modified as needed.
    Raises:
        FileNotFoundError: If the Kaggle API credentials file (`kaggle.json`) is missing.
        Exception: For any issues during data download, preprocessing, or augmentation.
    """


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
    train_samples = 33126
    test_samples = 10982
    
    if train_files < train_samples or test_files < test_samples:
        print("Downloading data...")
        get_data.run(train_samples, test_samples)
    else:
        print("Data already exists, skipping download")
    

    train_data, val_data, test_data = preprocess(smaller=True)

    num_augmentations = 3
    train_dataset_path = "data/train_images"
    synth_dir = "data/synthetic_images"
    
    # Create synthetic images directory if it doesn't exist
    os.makedirs(synth_dir, exist_ok=True)
    
    # Count existing synthetic images
    synthetic_files = len([f for f in os.listdir(train_dataset_path) if os.path.isfile(os.path.join(train_dataset_path, f))])
    expected_synthetic_files = len(train_data) * (num_augmentations + 1)

    
    synthetic_dataset = synth_data(train_data, num_augmentations)

    
    if True:
        print(f"Generating {expected_synthetic_files} synthetic images...")
        advanced_augmentation(train_dataset_path, synth_dir, train_data, augmentations_per_image=num_augmentations)
        
        # Move synthetic images to train images folder
        print("Moving synthetic images to train folder...")
        synth_image_files = [f for f in os.listdir(synth_dir) if os.path.isfile(os.path.join(synth_dir, f))]
        for file in synth_image_files:
            src_path = os.path.join(synth_dir, file)
            dst_path = os.path.join(train_dataset_path, file)
            if not os.path.exists(dst_path):  # Avoid overwriting existing files
                os.rename(src_path, dst_path)  # Move file (rename operation)
        print(f"Moved {len(synth_image_files)} synthetic images to training directory")

    else:
        print("Synthetic images already exist, skipping generation")


    
    train_data_small, val_data_small, test_data_small = one_hot(synthetic_dataset, val_data, test_data)

    model_results_small, model_small = base_model(train_data_small, val_data_small, train_dataset_path, data_type='small_small')


def synth_data(train_data, num_augmentations):
    synth_dataset = train_data.copy()
    for n in range(num_augmentations):
        synth_dataset_n = train_data.copy()
        synth_dataset_n['image_name'] = synth_dataset_n['image_name'] + f'_synth_{n}'
        synth_dataset = pd.concat([synth_dataset, synth_dataset_n])
    train_data = synth_dataset

    return train_data


def base_model(train_data, val_data, train_dataset_path, data_type):
    """
    Trains a base model using both tabular and image data.
    Args:
        train_data (pd.DataFrame): The training dataset containing tabular features and target labels.
        val_data (pd.DataFrame): The validation dataset containing tabular features and target labels.
        train_dataset_path (str): Path to the directory containing the training dataset, including images.
        data_type (str): A string identifier for the type of data being used (e.g., 'classification').
    Returns:
        tuple: A tuple containing:
            - model_results (dict): A dictionary containing training metrics and results.
            - model (torch.nn.Module): The trained PyTorch model.
    """

    # Define hyperparameters
    batch_size = 64
    num_epochs = 100
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = f'saved_models/base_{data_type}'

    tabular_features = train_data.columns[2:-1].to_list()
    train_loader, val_loader = create_dataloaders(train_data, val_data, train_dataset_path, 
                       img_id_column='image_name',
                       target_column='target',
                       tabular_columns=tabular_features,
                       batch_size=batch_size
                       )
   
    # Create directory for saving models if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    model = BaseModel(num_tabular_features=len(tabular_features), tabular_hidden_dims=[512, 128])

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    print(f"Training on {device} with {len(train_data)} training samples, {len(val_data)} validation samples")


    model_results = train(model, train_loader, val_loader, criterion, optimizer, 
                     device, num_epochs, model_dir, checkpoint_freq=1, save_best_only=True, 
                     early_stopping_patience=6, start_epoch=0)
    
    return model_results, model



def predict_with_model(model, test_data, test_dataset_path, device=None):
    """
    Makes predictions using a trained model on test data without target labels.
    
    Args:
        model: The trained model to use for predictions
        test_data: DataFrame containing test data (without target column)
        test_dataset_path: Path to test images directory
        device: Computation device (defaults to available GPU or CPU)
    
    Returns:
        predictions: List of model predictions
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Get tabular features (all columns except image_name)
    tabular_features = test_data.columns[1:].to_list()
    
    
    # Create test dataset and dataloader
    test_dataset = TestDataset(test_data, test_dataset_path, 'image_name', tabular_features)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    predictions = []
    image_ids = []
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for i, (images, tabular) in enumerate(test_loader):
            images = images.to(device)
            tabular = tabular.to(device)
            
            outputs = model(images, tabular)
   
            
            # Store predictions
            predictions.extend(outputs.cpu().numpy())
            
            # Store image IDs for this batch
            batch_indices = range(i * test_loader.batch_size, 
                                 min((i + 1) * test_loader.batch_size, len(test_data)))
            batch_img_ids = [test_data.iloc[idx]['image_name'] for idx in batch_indices]
            image_ids.extend(batch_img_ids)
    
    # Create a dictionary mapping image IDs to predictions
    results = {'image_name': image_ids, 'prediction': predictions}
    return pd.DataFrame(results)

if __name__ == "__main__":

    main()
