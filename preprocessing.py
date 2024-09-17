import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['tempo'] = pd.to_numeric(X['tempo'], errors='coerce')
        X['duration_ms'] = X['duration_ms'].replace(-1, np.nan)
        X['mode'] = X['mode'].replace(-1, np.nan)
        X['artist_name'] = X['artist_name'].replace('empty_field', 'missing')
        return X

class ArtistGenreEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.artist_genre_map_ = None

    def fit(self, X, y=None):
        if y is not None:
            temp_df = X.copy()
            temp_df['genre'] = y
            self.artist_genre_map_ = temp_df.groupby('artist_name')['genre'].agg(lambda x: x.value_counts().index[0]).to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        if self.artist_genre_map_ is not None:
            X['artist_genre'] = X['artist_name'].map(self.artist_genre_map_)
        else:
            X['artist_genre'] = 'unknown'
        return X

def create_preprocessing_pipeline():
    numeric_features = ['popularity', 'tempo', 'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
    categorical_features = ['artist_name', 'artist_genre', 'mode', 'key']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    # PCA to reduce dimensionality
    dimensionality_reduction = PCA(n_components=10)

    return Pipeline([
        ('missing_handler', MissingValueHandler()),
        ('artist_genre_encoder', ArtistGenreEncoder()),
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(score_func=f_classif, k=15)),
        ('PCA', dimensionality_reduction)
    ])

def load_and_preprocess_data(train_file, test_file):
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Split train data into features and target
    X_train = train_data.drop(['genre', 'track_id', 'instance_id', 'time_signature', 'track_name'], axis=1)
    y_train = train_data['genre']

    # Create and fit the preprocessing pipeline
    pipeline = create_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train, y_train)

    # Process test data
    X_test_processed = pipeline.transform(test_data)

    return X_train_processed, y_train, X_test_processed, pipeline

if __name__ == '__main__':
    # Load and preprocess data
    X_train, y_train, X_test, pipeline = load_and_preprocess_data('combined_data.csv', 'testing-data/testing-instances.csv')

    print(X_train.shape)
    print(y_train.shape)

    # convert from sparse matrix to DataFrame
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Save processed data
    pd.DataFrame(X_train).to_csv('data/X_train.csv', index=False)
    pd.DataFrame(y_train).to_csv('data/y_train.csv', index=False)
    pd.DataFrame(X_val).to_csv('data/X_val.csv', index=False)
    pd.DataFrame(y_val).to_csv('data/y_val.csv', index=False)
    pd.DataFrame(X_test).to_csv('testing-data/testing-instances-processed.csv', index=False)

    print('Preprocessing complete')