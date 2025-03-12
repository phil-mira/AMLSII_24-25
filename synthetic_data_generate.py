import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from diffusers import StableDiffusionPipeline, DDPMPipeline, DDIMScheduler
from diffusers.utils import load_image
import cv2


def advanced_augmentation(image_dir, output_dir, train_images, augmentations_per_image=1):
    """
    Apply more advanced augmentations using albumentations
    
    Args:
        images_dir(list): Directory of the input images
        output_dir (str): Directory to save augmented images
        augmentations_per_image (int): Number of augmented versions to create per original image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transforms with albumentations
    transform = A.Compose([
        A.Flip(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit=0.1)
        ], p=1.0),
        A.Resize(256, 256)
    ])
    
    synth_dataset = train_images.copy()
    for n in range(augmentations_per_image):
        synth_dataset_n = train_images.copy()
        synth_dataset_n['image_name'] = synth_dataset_n['image_name'] + f'_synth_{n}'
        synth_dataset = pd.concat([synth_dataset, synth_dataset_n])


    # Apply transforms and save images
    count = 0
    for i, img_file in enumerate(os.listdir(image_dir)):

        # Drop the file extension from img_file
        img_file_no_ext = os.path.splitext(img_file)[0]

        if img_file_no_ext in train_images['image_name'].values:
            
            img_path = os.path.join(image_dir, img_file)

            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_file}: {e}")

            # Convert PIL to numpy array
            img_np = np.array(img)
            
            for j in range(augmentations_per_image):
                augmented = transform(image=img_np)
                augmented_img = augmented['image']
                
                # Save image
                save_path = os.path.join(output_dir, f"{img_file_no_ext}_synth_{j}.jpg")
                cv2.imwrite(save_path, cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR))
                count += 1
        
    print(f"Generated {count} advanced augmented images in {output_dir}")

    return synth_dataset


def visualize_augmentations(original_img, num_samples=5):
    """
    Visualize different augmentations of a single image
    
    Args:
        original_img (PIL.Image): Original image to augment
        num_samples (int): Number of augmentations to display
    """

    
    # Advanced transforms
    advanced_transform = A.Compose([
        A.Flip(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit = 0.1, contrast_limit=0.1)
        ], p=1.0),
        A.Resize(256, 256)
    ])
    
    # Convert to numpy for albumentations
    original_np = np.array(original_img)
    
     # Create figure
    plt.figure(figsize=(12, 8))
    
    # Show original
    plt.subplot(1, num_samples+1, 1)
    plt.imshow(original_img)
    plt.title("Original")
    plt.axis("off")
  

    # Show advanced augmentations
    for i in range(num_samples):
        plt.subplot(1, num_samples+1, i+2)
        augmented = advanced_transform(image=original_np)
        aug_img = augmented['image']
        plt.imshow(aug_img)
        plt.title(f"Example {i+1}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    
    dataset_path = "data/train_images"
    output_dir = "data/synthetic_images"

    advanced_augmentation(dataset_path, output_dir, augmentations_per_image=1)