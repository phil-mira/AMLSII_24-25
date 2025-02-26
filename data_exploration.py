import pandas


def explore_data():

    train_data = pd.read_csv("data/train.csv", nrows=train_samples)
    test_data = pd.read_csv("data/test.csv", nrow=test_samples)
