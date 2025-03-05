import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



class MixedInputDataset(Dataset):
    """
    Dataset for handling both image and tabular data
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing image IDs and tabular features
    img_dir : str
        Directory containing the images
    img_id_column : str
        Column name in DataFrame that contains image IDs/filenames
    target_column : str
        Column name in DataFrame that contains target labels
    tabular_columns : list
        List of column names to use as tabular features
    transform : callable, optional
        Optional transform to be applied on images
    """
    def __init__(self, df, img_dir, img_id_column, target_column, tabular_columns, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.img_id_column = img_id_column
        self.target_column = target_column
        self.tabular_columns = tabular_columns
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image
        img_id = self.df.iloc[idx][self.img_id_column]
        img_path = os.path.join(self.img_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations to the image
        if self.transform:
            image = self.transform(image)
        
        # Get tabular features
        tabular_features = torch.tensor(
            self.df.iloc[idx][self.tabular_columns].values.astype(np.float32)
        )
        
        # Get label
        label = torch.tensor(self.df.iloc[idx][self.target_column], dtype=torch.long)
        
        return {
            'image': image, 
            'tabular': tabular_features, 
            'label': label,
            'img_id': img_id  # For debugging/tracking
        }



def create_dataloaders(train_df, val_df, img_dir, img_id_column, target_column, 
                       tabular_columns, batch_size, num_workers=4):
    """
    Create train and validation dataloaders
    
    Parameters:
    -----------
    train_df : DataFrame
        Training data
    val_df : DataFrame
        Validation data
    img_dir : str
        Directory containing images
    img_id_column : str
        Column name in CSV that contains image IDs/filenames
    target_column : str
        Column name in CSV that contains target labels
    tabular_columns : list
        List of column names to use as tabular features
    batch_size : int
        Batch size for dataloaders
    num_workers : int
        Number of worker processes for data loading
        
    Returns:
    --------
    tuple: (train_loader, val_loader)
        Training and validation data loaders
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets with appropriate transforms
    train_dataset = MixedInputDataset(
        train_df,
        img_dir,
        img_id_column,
        target_column,
        tabular_columns,
        transform=train_transform
    )
    
    val_dataset = MixedInputDataset(
        val_df,
        img_dir,
        img_id_column,
        target_column,
        tabular_columns,
        transform=val_transform
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
