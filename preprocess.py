import pandas as pd



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


# Group same people together so no data leakage to validation
def create_validation(train_data, val_size):
     
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
     

def preprocess():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    duplicates_df = pd.read_csv('data/2020_challenge_duplicates.csv')
    val_size = len(test_df)

    remove_duplicates(train_df, test_df, duplicates_df)
    train_df, val_df = create_validation(train_df, val_size)
    
    return train_df, val_df, test_df
