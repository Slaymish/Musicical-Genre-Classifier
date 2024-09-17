import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def load_data():
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    X_val = pd.read_csv('data/X_val.csv')
    y_val = pd.read_csv('data/y_val.csv').squeeze()
    return X_train, y_train, X_val, y_val

def create_stacking_model():
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
        ('gbt', GradientBoostingClassifier(n_estimators=100, random_state=0)),
        ('xgb', XGBClassifier(n_estimators=100))
    ]
    meta_model = LogisticRegression()
    return StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)

def create_simple_model():
    return RandomForestClassifier(n_estimators=100, random_state=0)

def create_fancy_model():
    return MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)

def create_SVM_model():
    return SVC()

def create_stacking_model2():
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=0)),
        ('gbt', MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000)),
        ('svm', SVC())
    ]
    meta_model = LogisticRegression()
    return StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)


def train_and_evaluate(X_train, y_train, X_val, y_val):
    #model = create_stacking_model()
    model = create_stacking_model2() 
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    return accuracy, model

def predict_and_save(model, X_test, file_path):
    predictions = model.predict(X_test)
    
    # Re-encode genre column (from numeric to categorical)
    #target_mapping = pd.Series(['Alternative', 'Blues', 'Classical', 'Comedy', 'Folk', 'Hip-Hop', 'Jazz', 'Opera', 'Pop', 'R&B'])
    #predictions = target_mapping[predictions]
    
    # Save the results
    testing_instances = pd.read_csv('testing-data/testing-instances.csv')
    results = pd.DataFrame({
        'instance_id': testing_instances['instance_id'],
        'genre': predictions
    })
    results.to_csv(file_path, index=False)
    print(f'Predictions saved to {file_path}')

def main():
    # Load data
    X_train, y_train, X_val, y_val = load_data()

    # Train and evaluate the model
    accuracy, model = train_and_evaluate(X_train, y_train, X_val, y_val)
    print(f'Stacking Model Accuracy: {accuracy}')

    # Load and predict on testing data
    X_test = pd.read_csv('testing-data/testing-instances-processed.csv')
    
    # Predict and save results
    predict_and_save(model, X_test, 'results.csv')
    predict_and_save(model, X_test, 'testing-data/predictions.csv')

if __name__ == '__main__':
    main()