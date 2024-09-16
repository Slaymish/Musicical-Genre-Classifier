import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis
from preprocessing import preprocess, encode_categorical_columns
def load_files():
    # Define the path to the directory containing the CSV files
    csv_dir = 'training-data/'

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(csv_dir + '*.csv')

    # Create an empty list to store the dataframes
    dfs = []

    # Iterate over each CSV file and load it into a dataframe
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    return dfs

def summary_stats(df):
    # Get num of instances and features
    num_instances, num_features = df.shape
    print(f"\nNumber of instances: {num_instances}")
    print(f"\nNumber of features: {num_features}")

    # Show ALL the data types
    print("\nData types:")
    print(df.dtypes)

    # Missing values
    print("\nMissing values:")

    # tempo has ? as missing
    # artist is empty field
    # duration_ms has -1 as missing

    # for tempo
    missing_tempo = df['tempo'].str.contains('\?').sum()
    print(f"tempo: {missing_tempo} missing values")

    # for artist
    missing_artist = df['artist_name'].isnull().sum()
    print(f"artist: {missing_artist} missing values")

def findTopCorrelations(df, target, n):
    # Ensure the target column exists in the DataFrame
    if target not in df.columns:
        raise KeyError(f"'{target}' column not found in DataFrame")

    numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32","int8"])

    numeric_df[target] = df[target]

    # Compute the correlation matrix
    correlation_matrix = numeric_df.corr()

    # Ensure the target column exists in the correlation matrix
    if target not in correlation_matrix.columns:
        raise KeyError(f"'{target}' column not found in correlation matrix")

    # Sort the correlation matrix by the target variable
    sorted_correlations = correlation_matrix[target].sort_values(ascending=False)

    # Remove the target variable from the list
    sorted_correlations = sorted_correlations.drop(target)

    # Get the top n features
    top_correlations = sorted_correlations.head(n)

    return top_correlations


def analyseFeatureDistributions(data: pd.DataFrame, topFeatures: list) -> pd.DataFrame:
    results = []

    for feature in topFeatures:
        binwidth = 2 * (data[feature].quantile(0.75) - data[feature].quantile(0.25)) / data[feature].size**(1/3)
        if binwidth == 0:
            binwidth = 1
        numBins = int(np.ceil((data[feature].max() - data[feature].min()) / binwidth))

        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], bins=numBins, binwidth=binwidth, kde=True)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f'images/hist_{feature}.png')

        skewness = skew(data[feature].dropna())
        kurt = kurtosis(data[feature].dropna())

        results.append({'Feature': feature, 'Skewness': skewness, 'Kurtosis': kurt})

    return pd.DataFrame(results)

def heatmap(df, n):
    corrlation_matrix = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corrlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('images/heatmap.png')

    # return the top n feature pairs 
    return corrlation_matrix.unstack().sort_values(ascending=False).drop_duplicates().head(n)




def main():
    # Load the CSV files into dataframes
    dfs = load_files()

    # Concatenate the dataframes into a single dataframe
    combined_df = pd.concat(dfs)

    # save the combined dataframe
    combined_df.to_csv('combined_data.csv', index=False)

    target = 'genre'

    # Display summary statistics
    summary_stats(combined_df)

    # Preprocess the data
    combined_df = encode_categorical_columns(combined_df)

    # find the top 5 correlations with the target variable
    top_correlations = findTopCorrelations(combined_df, target, 5)
    print(f"\nTop 5 correlations with '{target}':")
    print(top_correlations)


    # Analyse the distributions of the top 5 features
    top_features = top_correlations.index
    results = analyseFeatureDistributions(combined_df, top_features)

    print("\nSkewness and Kurtosis:")
    print(results)

    # Heatmap
    correlated_features_pairs = heatmap(combined_df, 5)
    print(f"\nTop {5} correlated features pairs:")
    print(correlated_features_pairs)

    # plots of the top 5 correlated features pairs
    for pair in correlated_features_pairs.index:
        feature1, feature2 = pair
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=combined_df)
        plt.title(f'{feature1} vs {feature2}')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.grid(True)
        plt.savefig(f'images/{feature1}_vs_{feature2}.png')


    # scatter plot of the top 5 correlated features
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature, y=target, data=combined_df)
        plt.title(f'{feature} vs {target}')
        plt.xlabel(feature)
        plt.ylabel(target)
        plt.grid(True)
        plt.savefig(f'images/{feature}_vs_{target}.png')


if __name__ == '__main__':
    main()