import pandas as pd
import numpy as np
from sklearn.ensemble import  RandomForestClassifier
from sklearn.feature_selection import RFE


def trainAndEval(X_train, y_train, X_test, y_test, model: object):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return accuracy, model

def main():
    target = 'genre'

    # Import preprocessing .csv files
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    accuracy, model = trainAndEval(X_train, y_train, X_test, y_test, model)

    print(f"Accuracy: {accuracy}")
    
    # Predict testing-data/testing-instances.csv
    testing_instances = pd.read_csv('testing-data/testing-instances.csv')
    testing_instances_processed = pd.read_csv('testing-data/testing-instances-processed.csv')

    target_mapping = pd.Series(['Alternative', 'Blues', 'Classical', 'Comedy', 'Folk', 'Hip-Hop','Jazz', 'Opera', 'Pop', 'R&B'])

    predictions = model.predict(testing_instances_processed)

    # Re-encode genre column (from numeric to categorical)
    predictions = target_mapping[predictions]

    print(predictions)

    # Save the results (in correct format for kaggle)
    # instance_id, genre (header)
    zip_data = zip(testing_instances['instance_id'], predictions)
    results = pd.DataFrame(zip_data, columns=['instance_id', 'genre'])
    results.to_csv('results.csv', index=False)
    



if __name__ == '__main__':
    main()