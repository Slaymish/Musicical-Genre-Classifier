import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer


def encode_categorical_columns(df):
    # Iterate over each column in the DataFrame
    for column in df.select_dtypes(include=['object', 'category']).columns:
        # Encode the column as a categorical type
        df[column] = df[column].astype('category')

        # Handle NaN values for categorical columns
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

        # Handle NaN values
        if df.isnull().sum() > 0:
            df = df.cat.add_categories('NaN')
            df = df.fillna('NaN')

        # Replace the column with the encoded values
        df = df.cat.codes

    return df

def drop_columns(df, columns):
    # Ensure the columns are in the DataFrame
    df = pd.DataFrame(df)  # In case it's an array, ensure it's a DataFrame
    for column in columns:
        if column not in df.columns:
            raise KeyError(f"'{column}' column not found in DataFrame")
    
    # Drop the columns from the DataFrame
    df = df.drop(columns=columns)
    return df


def standardize(X_train, X_test):
    # Initialize the standard scaler
    scaler = StandardScaler()

    # Fit the scaler on the training data
    scaler.fit(X_train)

    # Transform the training and testing data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

def set_missing_data_to_nan(df):
    # Replace placeholders with NaN
    df['tempo'] = df['tempo'].replace('?', np.nan)
    df['duration_ms'] = df['duration_ms'].replace(-1, np.nan)
    return df

def engineer_features(df):
    df['tempo_category'] = pd.cut(df['tempo'], bins=[0, 90, 120, float('inf')], labels=['slow', 'medium', 'fast'])
    df['popularity_bin'] = pd.qcut(df['popularity'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    df['energy_loudness'] = df['energy'] * df['loudness']
    return df

def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X[selected_features]


def splitData(df, target):
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def handle_missing_values(df):
    df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
    df['tempo'].fillna(df['tempo'].median(), inplace=True)
    df['duration_ms'].fillna(df['duration_ms'].median(), inplace=True)
    return df

def impute_missing_values(df, imputer=None):
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(df)
    return pd.DataFrame(imputer.transform(df), columns=df.columns), imputer

def preprocess(X_train, y_train, X_test, y_test):
    # Handle missing data
    X_train = set_missing_data_to_nan(X_train)
    X_test = set_missing_data_to_nan(X_test)

    # Engineer features
    #X_train = engineer_features(X_train)
    #X_test = engineer_features(X_test)

    # Encode the categorical columns
    X_train = encode_categorical_columns(X_train)
    X_test = encode_categorical_columns(X_test)

    # Get column names before converting to numpy arrays
    original_columns = X_train.columns

    # Impute missing values (use median strategy for numeric features)
    X_train, imputer = impute_missing_values(X_train)
    X_test = imputer.transform(X_test)

    # Encode the target column
    y_train = encode_target_column(y_train)
    y_test = encode_target_column(y_test)

    # Drop ignored columns
    global ignored_columns
    ignored_columns = ['track_id', 'instance_id', 'time_signature', 'track_name', 'artist_name']
    X_train = pd.DataFrame(X_train, columns=original_columns)  # Convert back to DataFrame after imputation
    X_test = pd.DataFrame(X_test, columns=original_columns)
    X_train = drop_columns(X_train, ignored_columns)
    X_test = drop_columns(X_test, ignored_columns)

    # Feature selection (optional, depends on your goal)
    #X_train = select_features(X_train, y_train)
    #X_test = X_test[X_train.columns]

    # Standardize the data
    X_train, X_test, scaler = standardize(X_train, X_test)

    # Convert standardized data back to DataFrame (to maintain consistency)
    X_train = pd.DataFrame(X_train, columns=[col for col in original_columns if col not in ignored_columns])
    X_test = pd.DataFrame(X_test, columns=[col for col in original_columns if col not in ignored_columns])

    # Return preprocessed data
    return X_train, X_test, pd.DataFrame(y_train), pd.DataFrame(y_test), scaler, imputer




if __name__ == '__main__':
    df = pd.read_csv('combined_data.csv')
    target = 'genre'

    # split
    X_train, X_val, y_train, y_val = splitData(df, target)

    # preprocess
    X_preprocessed_train, X_preprocessed_val, y_preprocessed_train, y_preprocessed_val,scaler,imputer = preprocess(X_train, y_train, X_val, y_val)

    # save preprocessed data
    X_preprocessed_train.to_csv('data/X_train.csv', index=False)
    y_preprocessed_train.to_csv('data/y_train.csv', index=False)

    X_preprocessed_val.to_csv('data/X_val.csv', index=False)
    y_preprocessed_val.to_csv('data/y_val.csv', index=False)

    # Process testing-instances.csv
    testing_instances = pd.read_csv('testing-data/testing-instances.csv')
    testing_instances = set_missing_data_to_nan(testing_instances)
    testing_instances = encode_categorical_columns(testing_instances)
    testing_instances, _ = impute_missing_values(testing_instances, imputer)
    testing_instances = drop_columns(testing_instances, ignored_columns)
    testing_instances = pd.DataFrame(scaler.transform(testing_instances), columns=testing_instances.columns)
    testing_instances.to_csv('testing-data/testing-instances-processed.csv', index=False)


    print('Preprocessing complete')