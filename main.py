import os
import get_data
import json

import data_exploration
import get_data

import kaggle
import pandas as pd


def main():

    # Navigate to .kaggle directory and load credentials
    kaggle_path = os.path.join(os.path.expanduser('~'), '.kaggle')
    if not os.path.exists(kaggle_path):
        os.makedirs(kaggle_path)

    # Load kaggle.json file
    with open(os.path.join(kaggle_path, 'kaggle.json')) as f:
        api_token = json.load(f)

    train_samples = 50
    test_samples = 10
    get_data.run(train_samples, test_samples)


if __name__ == "__main__":

    main()
