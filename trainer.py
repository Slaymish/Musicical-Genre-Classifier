import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def feature_selection(X_train, y_train, model, n_features=10):
    """Apply Recursive Feature Elimination to select top n features."""
    selector = RFE(model, n_features_to_select=n_features, step=1)
    selector = selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    return X_train_selected, selector

def hyperparameter_tuning(X_train, y_train, model, param_grid):
    """Perform GridSearchCV for hyperparameter tuning."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def ensemble_voting(models):
    """Create an ensemble using VotingClassifier."""
    voting_clf = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
    return voting_clf

def stacking(models, final_estimator):
    """Create a stacking classifier."""
    stack_clf = StackingClassifier(estimators=models, final_estimator=final_estimator, n_jobs=-1, cv=5)
    return stack_clf

def train_and_eval(X_train, y_train, X_test, y_test, model):
    """Train model and evaluate accuracy."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, model

def neural_network(X_train, y_train, X_test, y_test):
    """Train a neural network model using scikit-learn."""
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0)
    accuracy, model = train_and_eval(X_train, y_train, X_test, y_test, model)
    return accuracy, model

def main():
    target = 'genre'

    # Import preprocessing .csv files
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()

    X_val = pd.read_csv('data/X_val.csv')
    y_val = pd.read_csv('data/y_val.csv').squeeze()

    # Define base models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
    gbt_model = HistGradientBoostingClassifier()
    xgb_model = XGBClassifier(n_estimators=100)

    # Base models for stacking
    base_models = [
        ('rf', rf_model),
        ('gbt', gbt_model),
        ('xgb', xgb_model)
    ]

    # Meta learner (stacking)
    meta_model = LogisticRegression()

    # Build stacking classifier
    stack_model = stacking(base_models, meta_model)

    # Train and evaluate the stacking model
    accuracy, stack_model = train_and_eval(X_train, y_train, X_val, y_val, stack_model)
    print(f'Stacking Model Accuracy: {accuracy}')

    # Compare to neural network model
    nn_params = {
        'hidden_layer_sizes': [(100, 50), (100, 100), (50, 50)],
        'max_iter': [1000, 2000]
    }

    """ print('Training Neural Network Model...')
    nn_model = MLPClassifier(random_state=0)
    nn_model = hyperparameter_tuning(X_train, y_train, nn_model, nn_params)
    print('Evaluating Neural Network Model...')
    print('Best Parameters:', nn_params)
    accuracy, nn_model = train_and_eval(X_train, y_train, X_val, y_val, nn_model) """

    # Predict on testing data
    testing_instances_processed = pd.read_csv('testing-data/testing-instances-processed.csv')

    # Make predictions with the stacked model
    predictions = stack_model.predict(testing_instances_processed)

    # Re-encode genre column (from numeric to categorical)
    target_mapping = pd.Series(['Alternative', 'Blues', 'Classical', 'Comedy', 'Folk', 'Hip-Hop', 'Jazz', 'Opera', 'Pop', 'R&B'])
    predictions = target_mapping[predictions]

    # Save the results
    testing_instances = pd.read_csv('testing-data/testing-instances.csv')
    zip_data = zip(testing_instances['instance_id'], predictions)
    results = pd.DataFrame(zip_data, columns=['instance_id', 'genre'])
    results.to_csv('results.csv', index=False)

    # Save predictions
    prediction_output = pd.DataFrame(predictions, columns=['Predicted Genre'])
    prediction_output.to_csv('testing-data/predictions.csv', index=False)
    print('Predictions saved to testing-data/predictions.csv')



if __name__ == '__main__':
    main()
