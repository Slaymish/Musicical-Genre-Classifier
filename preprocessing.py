import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

def encode_categorical_columns(df):
    # Iterate over each column in the DataFrame
    for column in df.select_dtypes(include=['object', 'category']).columns:
        # Encode the column as a categorical type
        df[column] = df[column].astype('category')

        # skip over NaN values
        if df[column].isnull().sum() > 0:
            df[column] = df[column].cat.add_categories('NaN')
            df[column] = df[column].fillna('NaN')

        # Replace the column with the encoded values
        df[column] = df[column].cat.codes

    return df

def encode_target_column(df):
    if isinstance(df, pd.Series):
        # Encode the column as a categorical type
        df = df.astype('category')

        # skip over NaN values
        if df.isnull().sum() > 0:
            df = df.cat.add_categories('NaN')
            df = df.fillna('NaN')

        # Replace the column with the encoded values
        df = df.cat.codes
        
    return df

def drop_columns(df, columns):
    # Ensure the columns are in the DataFrame
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"'{column}' column not found in DataFrame")
    
    # Drop the columns from the DataFrame
    df = df.drop(columns=columns)
    return df

def one_hot_encode(df, columns):
    # Ensure the columns are in the DataFrame
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"'{column}' column not found in DataFrame")
    
    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=columns)
    return df


def standardize(df):
    global scaler
    # Standardize the data
    scaler = StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df, scaler

def setMissingDataToNaN(df):
    # tempo has ? as missing
    # artist is empty field
    # duration_ms has -1 as missing

    # for tempo
    df['tempo'] = df['tempo'].replace('?', np.nan)

    # for duration_ms
    df['duration_ms'] = df['duration_ms'].replace(-1, np.nan)

    return df

def splitData(df, target):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test



def preprocess(X_train, y_train, X_test, y_test):
    # Handle Missing data
    X_train, X_test = setMissingDataToNaN(X_train), setMissingDataToNaN(X_test)

    # Encode the categorical columns
    X_train = encode_categorical_columns(X_train)
    X_test = encode_categorical_columns(X_test)

    y_train = encode_target_column(y_train)
    y_test = encode_target_column(y_test)

    # for y, one hot encode
    #y_train = pd.get_dummies(y_train)
    #y_test = pd.get_dummies(y_test)


    # Drop the ignored columns
    global ignored_columns
    ignored_columns = ['track_id', 'instance_id','time_signature','track_name','artist_name']
    X_train = drop_columns(X_train, ignored_columns)
    X_test = drop_columns(X_test, ignored_columns)

    # Standardize the data
    #df, scaler = standardize(df)

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    df = pd.read_csv('combined_data.csv')
    target = 'genre'

    # split
    X_train, X_val, y_train, y_val = splitData(df, target)

    # preprocess
    X_preprocessed_train, X_preprocessed_val, y_preprocessed_train, y_preprocessed_val = preprocess(X_train, y_train, X_val, y_val)

    # save preprocessed data
    X_preprocessed_train.to_csv('data/X_train.csv', index=False)
    y_preprocessed_train.to_csv('data/y_train.csv', index=False)

    X_preprocessed_val.to_csv('data/X_val.csv', index=False)
    y_preprocessed_val.to_csv('data/y_val.csv', index=False)

    # process testing-instances.csv
    testing_instances = pd.read_csv('testing-data/testing-instances.csv')
    processed_testing_instances = setMissingDataToNaN(testing_instances)
    processed_testing_instances = encode_categorical_columns(processed_testing_instances)
    processed_testing_instances = drop_columns(processed_testing_instances, ignored_columns)
    processed_testing_instances.to_csv('testing-data/testing-instances-processed.csv', index=False)

    print('Preprocessing complete')