import pandas as pd
from sklearn.model_selection import train_test_split



# Remove duplicates
def remove_duplicates(train_data, test_data, duplicates_data):    
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
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    duplicates_df = pd.read_csv('data/2020_challenge_duplicates.csv')
    val_size = len(test_df)

    remove_duplicates(train_df, test_df, duplicates_df)
    
    train_df = train_df.dropna()
    test_df = test_df.dropna()
   
    if smaller == True:
        train_df = create_smaller_dataset(train_df)
        train_df, val_df = create_validation_small(train_df)
        
    else:
        train_df, val_df = create_validation_all(train_df, val_size)

    
    
    return train_df, val_df, test_df
