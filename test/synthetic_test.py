import unittest
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 


from synthetic_data_generate import advanced_augmentation, visualize_augmentations

class TestSyntheticDataGenerate(unittest.TestCase):
    def setUp(self):
        # Create temporary test directories
        self.test_input_dir = "test_input_images"
        self.test_output_dir = "test_output_images"
        os.makedirs(self.test_input_dir, exist_ok=True)
        
        # Create sample test images
        self.test_image_names = ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"]
        for img_name in self.test_image_names:
            # Create a simple test image (100x100 RGB)
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(os.path.join(self.test_input_dir, img_name), img)
        
        # Create sample dataframe with image names
        self.train_images = pd.DataFrame({
            'image_name': self.test_image_names[:2]  # Only include first 2 images
        })

    def tearDown(self):
        # Clean up test directories
        if os.path.exists(self.test_input_dir):
            shutil.rmtree(self.test_input_dir)
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def test_advanced_augmentation_output_format(self):
        """Test that advanced augmentation creates properly formatted images"""
        # Run the augmentation function
        advanced_augmentation(
            self.test_input_dir, 
            self.test_output_dir, 
            self.train_images, 
            augmentations_per_image=2
        )
        
        # Check output directory was created
        self.assertTrue(os.path.exists(self.test_output_dir))
        
        # Check number of augmented images (2 images * 2 augmentations)
        output_files = os.listdir(self.test_output_dir)
        self.assertEqual(len(output_files), 4)
        
        # Check all output files are valid images with correct dimensions
        for file_name in output_files:
            file_path = os.path.join(self.test_output_dir, file_name)
            self.assertTrue(os.path.exists(file_path))
            
            # Check file is a valid image
            img = cv2.imread(file_path)
            self.assertIsNotNone(img)
            
            # Check dimensions (should be 256x256 per the function)
            self.assertEqual(img.shape[0], 256)
            self.assertEqual(img.shape[1], 256)
            self.assertEqual(img.shape[2], 3)  # RGB channels


if __name__ == "__main__":
    unittest.main()