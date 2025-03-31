import os
import get_data

import get_data
from preprocess import preprocess, one_hot
from synthetic_data_generate import advanced_augmentation
from data_utils import create_dataloaders
from train_validate import train

from models import BaseModel, SexBasedModel, BrightnessBasedModel

import pandas as pd
import torch



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


    train_dataset_path = "data/train_images"
    synth_dir = "data/synthetic_images"

    data_model = [
     ['base_all', BaseModel(num_tabular_features=9), 0, False],
     ['base_small', BaseModel(num_tabular_features=9), 0, True],
     ['base_small_3xaug', BaseModel(num_tabular_features=9), 3, True],
     ['base_small_5xaug', BaseModel(num_tabular_features=9), 5, True],
     ['sex_based_all', SexBasedModel(num_tabular_features=9), 0, False],
     ['sex_based_small', SexBasedModel(num_tabular_features=9), 0, True],
     ['sex_based_small_3xaug', SexBasedModel(num_tabular_features=9), 3, True],
     ['sex_based_small_5xaug', SexBasedModel(num_tabular_features=9), 5, True],
     ['brightness_based_all', BrightnessBasedModel(num_tabular_features=9), 0, False],
     ['brightness_based_small', BrightnessBasedModel(num_tabular_features=9), 0, True],
     ['brightness_based_small_3xaug', BrightnessBasedModel(num_tabular_features=9), 3, True],
     ['brightness_based_small_5xaug', BrightnessBasedModel(num_tabular_features=9), 5, True]
    ]
    
    for data, model, num_augmentations, small in data_model:

        train_data, val_data, test_data = preprocess(smaller=small)
    

        # Create synthetic images directory if it doesn't exist
        os.makedirs(synth_dir, exist_ok=True)

        synthetic_dataset = synth_data(train_data, num_augmentations)

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

        train_data, val_data, test_data = one_hot(synthetic_dataset, val_data, test_data)

        run_model(train_data, val_data, train_dataset_path, model, data_type=data)




def synth_data(train_data, num_augmentations):
    """
    Synthesize additional data by creating copies of the training dataset.
    This function creates multiple copies of the training data to augment the dataset,
    with each copy having a modified image name to indicate it's synthetic data.
    Parameters:
    ----------
    train_data : pandas.DataFrame
        The original training dataset to be augmented.
        Expected to have an 'image_name' column.
    num_augmentations : int
        The number of synthetic copies to create.
    Returns:
    -------
    pandas.DataFrame
        An augmented dataset containing the original data plus 'num_augmentations'
        copies with modified image names.
    """

    synth_dataset = train_data.copy()
    for n in range(num_augmentations):
        synth_dataset_n = train_data.copy()
        synth_dataset_n['image_name'] = synth_dataset_n['image_name'] + f'_synth_{n}'
        synth_dataset = pd.concat([synth_dataset, synth_dataset_n])
    train_data = synth_dataset

    return train_data


def run_model(train_data, val_data, train_dataset_path, model, data_type):
    """
    Trains a model using both tabular and image data.
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
    num_epochs = 10
    learning_rate = 0.00001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = f'saved_models/{data_type}'

    tabular_features = train_data.columns[2:-1].to_list()
    train_loader, val_loader = create_dataloaders(train_data, val_data, train_dataset_path, 
                       img_id_column='image_name',
                       target_column='target',
                       tabular_columns=tabular_features,
                       batch_size=batch_size
                       )
   
    # Create directory for saving models if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)


    # Define loss function and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Learning rate scheduler - Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # Total number of epochs
        eta_min=1e-8       # Minimum learning rate
    )


    print(f"Training on {device} with {len(train_data)} training samples, {len(val_data)} validation samples")


    model_results = train(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                     device, num_epochs, model_dir, checkpoint_freq=1, save_best_only=True, 
                     early_stopping_patience=3, start_epoch=0)
    
    return model_results, model




if __name__ == "__main__":

    main()
