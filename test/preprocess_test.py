import unittest
import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # nopep8


from preprocess import preprocess
from preprocess import remove_duplicates
from preprocess import create_validation

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.read_csv('data/train.csv')
        self.test_df = pd.read_csv('data/test.csv')
        self.duplicates_df = pd.read_csv('data/2020_challenge_duplicates.csv')
        self.val_size = len(self.test_df)

    def test_remove_duplicates(self):
        """Test duplicates are removed correctly."""
        
        # Make copies to avoid modifying original data
        train_copy = self.train_df.copy()
        test_copy = self.test_df.copy()
        
        # Get counts before
        train_before = len(train_copy)
        test_before = len(test_copy)
        
        # Run function
        remove_duplicates(train_copy, test_copy, self.duplicates_df)
        
        # Check that duplicates were removed
        self.assertLess(len(train_copy), train_before)
        self.assertLess(len(test_copy), test_before)
        
        # Check no duplicates remain
        train_dups = set(train_copy['image_name']) & set(self.duplicates_df['ISIC_id_paired'])
        test_dups = set(test_copy['image_name']) & set(self.duplicates_df['ISIC_id_paired'])
        self.assertEqual(len(train_dups), 0)
        self.assertEqual(len(test_dups), 0)

    def test_create_validation(self):
        """Test validation set creation."""
        
        # Make a copy to avoid modifying original data
        train_copy = self.train_df.copy()
        
        # Run function
        train_result, val_result = create_validation(train_copy, self.val_size)
        
        # Check sizes
        self.assertGreater(len(train_result), 0)
        self.assertGreater(len(val_result), 0)
        self.assertEqual(len(train_result) + len(val_result), len(train_copy))
        
        # Check no patient overlap between train and validation
        train_patients = set(train_result['patient_id'])
        val_patients = set(val_result['patient_id'])
        self.assertEqual(len(train_patients & val_patients), 0)

    def test_preprocess(self):
        """Test the complete preprocessing pipeline."""
        # Run function
        train_result, val_result, test_result = preprocess()
        
        # Check all dataframes exist with data
        self.assertIsInstance(train_result, pd.DataFrame)
        self.assertIsInstance(val_result, pd.DataFrame)
        self.assertIsInstance(test_result, pd.DataFrame)
        self.assertGreater(len(train_result), 0)
        self.assertGreater(len(val_result), 0)
        self.assertGreater(len(test_result), 0)
        
        # Check no duplicates from duplicates_df exist in the results
        train_dups = set(train_result['image_name']) & set(self.duplicates_df['ISIC_id_paired'])
        val_dups = set(val_result['image_name']) & set(self.duplicates_df['ISIC_id_paired'])
        test_dups = set(test_result['image_name']) & set(self.duplicates_df['ISIC_id_paired'])
        
        self.assertEqual(len(train_dups), 0)
        self.assertEqual(len(val_dups), 0)
        self.assertEqual(len(test_dups), 0)