import sys
import unittest
import os
import pandas as pd
from PIL import Image
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # nopep8


from get_data import get_files_process


class TestGetData(unittest.TestCase):
    def setUp(self):
        self.competition_name = "siim-isic-melanoma-classification"
        self.train_file = "train.csv"
        self.test_file = "test.csv"
        self.samples = 5
        get_files_process(self.competition_name, self.train_file, self.samples)

    def test_csv_file_format(self):
        df = pd.read_csv(f"data/{self.train_file}")
        self.assertIn("image_name", df.columns)

    def test_image_files_exist(self):
        image_folder = f"data/train_images/"
        for image_name in pd.read_csv(f"data/{self.train_file}", nrows=self.samples)["image_name"]:
            self.assertTrue(os.path.exists(f"{image_folder}{image_name}.jpg"))

    def test_image_format(self):
        image_folder = f"data/train_images/"
        sample_image = pd.read_csv(
            f"data/{self.train_file}", nrows=1)["image_name"].iloc[0]
        with Image.open(f"{image_folder}{sample_image}.jpg") as img:
            self.assertEqual(img.format, "JPEG")

    def test_image_number(self):
        image_folder = f"data/train_images/"
        image_count = len([f for f in os.listdir(
            image_folder) if f.endswith('.jpg')])
        self.assertEqual(image_count, self.samples)

    def test_image_dimensions(self):
        image_folder = f"data/train_images/"
        for image_file in os.listdir(image_folder):
            if image_file.endswith('.jpg'):
                with Image.open(f"{image_folder}{image_file}") as img:
                    self.assertEqual(img.size, (256, 256), f"Image {image_file} is not 256x256 pixels")

    def tearDown(self):
        # Clean up test files
        if os.path.exists("data"):
            shutil.rmtree("data")


if __name__ == "__main__":
    unittest.main()
