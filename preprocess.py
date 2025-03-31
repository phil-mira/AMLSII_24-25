import pandas as pd
from sklearn.model_selection import train_test_split



# Remove duplicates
def remove_duplicates(train_data, test_data, duplicates_data):    
    """
    Remove duplicate images from train and test datasets based on duplicates data.

    This function identifies and removes duplicate images from both train and test datasets
    to prevent data leakage and ensure model evaluation integrity. It processes a duplicates
    dataframe that identifies paired images that are duplicates of each other, and removes
    the second image of each pair.

    Parameters:
    ----------
    train_data : pandas.DataFrame
        Training dataset containing image information
    test_data : pandas.DataFrame
        Test dataset containing image information
    duplicates_data : pandas.DataFrame
        DataFrame containing information about duplicate image pairs
        Expected to have columns 'partition' and 'ISIC_id_paired'

    Returns:
    -------
    None
        The function modifies train_data and test_data in place
    """
    # Create a set of duplicate image IDs to remove
    train_duplicate_ids = set()
    test_duplicate_ids = set()
    for num, val in enumerate(duplicates_data['partition']):
        # Keep the first image, remove the duplicate
        if val == 'train':
            train_duplicate_ids.add(duplicates_data.iloc[num]['ISIC_id_paired'])
        else: 
            test_duplicate_ids.add(duplicates_data.iloc[num]['ISIC_id_paired'])

    # Remove identified duplicates in place
    train_data.drop(train_data[train_data['image_name'].isin(train_duplicate_ids)].index, inplace=True)
    test_data.drop(test_data[test_data['image_name'].isin(test_duplicate_ids)].index, inplace=True)


def create_smaller_dataset(train_data):
    """
    Create a smaller, balanced dataset from the original training data.

    This function creates a reduced dataset by including all positive samples (melanoma)
    and only two samples per patient from the negative class (non-melanoma). This helps 
    address class imbalance and creates a more manageable dataset for faster experimentation
    while preserving the diversity of patients.

    Parameters:
    ----------
    train_data : pandas.DataFrame
        The original training dataset containing at least 'patient_id', 'target',
        and 'image_name' columns

    Returns:
    -------
    pandas.DataFrame
        A smaller, more balanced dataset containing all positive samples and
        a limited number of negative samples per patient
    """
    # Count samples per patient
    patient_sample_counts = train_data['patient_id'].value_counts().reset_index()
    patient_sample_counts.columns = ['patient_id', 'sample_count']

    # Sort patients by their sample count (ascending)
    patient_sample_counts = patient_sample_counts.sort_values('sample_count')

    duplicated_data = train_data.sort_values(['patient_id'])

    # Get all positive samples
    positive_samples = train_data[train_data['target'] == 1]

    # Remove positive samples from duplicated data
    duplicated_data = duplicated_data[~duplicated_data['image_name'].isin(positive_samples['image_name'])]

    # Group by patient_id and sample two rows per patient
    sampled_data = duplicated_data.groupby('patient_id').apply(lambda x: x.sample(n=2, replace=True)).reset_index(drop=True)

    combined_data = pd.concat([positive_samples, sampled_data], ignore_index=True)

    return combined_data


def create_validation_all(train_data, val_size):
    """
    Creates a validation set from the training data based on patient IDs to avoid data leakage.
    This function splits the data in a way that ensures patients in the validation set
    are completely separate from those in the training set. It sorts the data by patient_id
    and then identifies a split point that does not break in the middle of a patient's data.
    Parameters:
    -----------
    train_data : pandas.DataFrame
        The training data containing at least 'patient_id' column.
    val_size : int
        The approximate number of samples desired in the validation set.
    Returns:
    --------
    tuple of pandas.DataFrame
        A tuple containing (train_data, val_data) where:
        - train_data: DataFrame containing the training samples
        - val_data: DataFrame containing the validation samples
    Notes:
    ------
    The actual size of the validation set may be slightly larger than val_size
    to ensure that all data for a given patient stays together.
    """

     
    train_data.sort_values(by = ['patient_id'], inplace = True)
    id_to_split = round(len(train_data)-val_size)

    #check the values after to ensure that it is not the same patient
    split_val = train_data.iloc[id_to_split]['patient_id']
    next_val =  train_data.iloc[id_to_split+1]['patient_id']
    while split_val == next_val:
        id_to_split += 1
        split_val = train_data.iloc[id_to_split]['patient_id']
        next_val =  train_data.iloc[id_to_split+1]['patient_id']
         
    id_to_split += 1
    val_data = train_data.iloc[id_to_split:]
    train_data = train_data.iloc[:id_to_split]


    return train_data, val_data


def create_validation_small(train_data):
    """
    Create a validation split from a smaller dataset while preserving patient separation.

    This function creates a validation set by splitting patients rather than individual 
    samples to prevent data leakage. It ensures that samples from the same patient are 
    either all in the training set or all in the validation set. The function also 
    performs stratified sampling based on the melanoma target, ensuring balanced class 
    distribution in both splits.

    Parameters:
    ----------
    train_data : pandas.DataFrame
        The input training data containing 'patient_id' and 'target' columns

    Returns:
    -------
    tuple of pandas.DataFrame
        A tuple containing (train_data, val_data) with samples split by patient ID
        while maintaining class balance
    """
    # Stratified sampling based on target (melanoma) while respecting patient IDs
    # Group by patient_id and get the target value for each patient
    patient_targets = train_data.groupby('patient_id')['target'].mean().reset_index()
    # If a patient has any positive samples, we'll consider them positive (mean > 0)
    patient_targets['target_binary'] = (patient_targets['target'] > 0).astype(int)

    # Split patients, not individual samples, stratifying by target
    train_patients, val_patients = train_test_split(
        patient_targets['patient_id'],
        test_size=300,
        stratify=patient_targets['target_binary'],
        random_state=42
    )

    # Split the data based on patient assignment
    val_data = train_data[train_data['patient_id'].isin(val_patients)]
    train_data = train_data[train_data['patient_id'].isin(train_patients)]


    return train_data, val_data



def one_hot(train_data, val_data, test_data):
    """
    Apply one-hot encoding to categorical features in the dataset.
    
    This function converts categorical variables into a one-hot encoded format,
    ensures consistency of columns across all datasets, and reorders columns
    to place the target variable at the end.
    
    Parameters:
    ----------
    train_data : pandas.DataFrame
        Training dataset containing categorical features to encode
    val_data : pandas.DataFrame
        Validation dataset containing categorical features to encode
    test_data : pandas.DataFrame
        Test dataset containing categorical features to encode
        
    Returns:
    -------
    tuple of pandas.DataFrame
        A tuple containing (train_data, val_data, test_data) with
        categorical features one-hot encoded and consistent columns
    """
      # One-hot encode categorical features
    print("Applying one-hot encoding to categorical features...")

    # Identify categorical columns (explicitly defining them for clarity)
    categorical_columns = ['sex', 'anatom_site_general_challenge']

    # Apply one-hot encoding
    train_data_encoded = pd.get_dummies(train_data, columns=categorical_columns, drop_first=False)
    val_data_encoded = pd.get_dummies(val_data, columns=categorical_columns, drop_first=False)
    test_data_encoded = pd.get_dummies(test_data, columns=categorical_columns, drop_first=False)

    # Ensure all datasets have the same columns (in case some categories only appear in certain splits)
    all_columns = set(train_data_encoded.columns).union(set(val_data_encoded.columns)).union(set(test_data_encoded.columns))
    for column in all_columns:
        if column not in train_data_encoded:
            train_data_encoded[column] = 0
        if column not in val_data_encoded:
            val_data_encoded[column] = 0
        if column not in test_data_encoded:
            test_data_encoded[column] = 0

    # Replace original dataframes with encoded versions
    train_data = train_data_encoded
    val_data = val_data_encoded
    test_data = test_data_encoded

    # Drop the diagnosis and benign_malignant columns
    train_data = train_data.drop(['diagnosis', 'benign_malignant'], axis=1)
    val_data = val_data.drop(['diagnosis', 'benign_malignant'], axis=1)

    # Move target column to the end
    target_col = 'target'
    for df in [train_data, val_data, test_data]:
        target_values = df[target_col].copy()
        df.drop(target_col, axis=1, inplace=True)
        df[target_col] = target_values

    print(f"After encoding: train_data has {train_data.shape[1]} features, val_data has {val_data.shape[1]} features")
    
    return train_data, val_data, test_data

     

def preprocess(smaller=False):
    """
    Preprocess the ISIC skin lesion dataset for model training.

    This function loads training and test data, removes duplicate images, handles missing values,
    and creates training/validation splits. It can generate either a full dataset or a smaller
    balanced dataset for faster experimentation.

    Parameters:
    ----------
    smaller : bool, default=False
        If True, creates a smaller balanced dataset with all positive samples and 
        2 samples per patient from negative class. If False, uses the full dataset.

    Returns:
    -------
    tuple of pandas.DataFrame
        A tuple containing (train_df, val_df, test_df) - preprocessed dataframes 
        for training, validation, and testing respectively.
    """
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    duplicates_df = pd.read_csv('data/2020_challenge_duplicates.csv')
    val_size = len(train_df)*0.2

    remove_duplicates(train_df, test_df, duplicates_df)
    
    train_df = train_df.dropna()
    test_df = test_df.dropna()
   
    if smaller == True:
        train_df = create_smaller_dataset(train_df)
        train_df, val_df = create_validation_small(train_df)
        
    else:
        train_df, val_df = create_validation_all(train_df, val_size)

    
    
    return train_df, val_df, test_df
