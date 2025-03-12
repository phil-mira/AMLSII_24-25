import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image
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
    """
    def __init__(self, df, img_dir, img_id_column, target_column, tabular_columns):
        self.df = df
        self.img_dir = img_dir
        self.img_id_column = img_id_column
        self.target_column = target_column
        self.tabular_columns = tabular_columns

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image
        img_id = self.df.iloc[idx][self.img_id_column]
        img_path = os.path.join(self.img_dir, img_id)
        img_path = f"{img_path}.jpg"
        try:
            # Load image using PIL and convert to tensor
            image = Image.open(img_path).convert('RGB')
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder image if loading fails
            image = torch.zeros((3, 224, 224), dtype=torch.float32)
            
        # Get tabular features
        tabular_features = torch.tensor(
            self.df[self.tabular_columns].iloc[idx].values.astype(np.float32)
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
        
    Returns:
    --------
    tuple: (train_loader, val_loader)
        Training and validation data loaders
    """
    
    # Create datasets with appropriate transforms
    train_dataset = MixedInputDataset(
        train_df,
        img_dir,
        img_id_column,
        target_column,
        tabular_columns,
    )
    
    val_dataset = MixedInputDataset(
        val_df,
        img_dir,
        img_id_column,
        target_column,
        tabular_columns,
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, val_loader
