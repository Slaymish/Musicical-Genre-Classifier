# Music genre classifier

A classifier for music genres, trained on the Kaggle genre dataset: ten genres, 5,000 tracks each, 50,000 rows. The preprocessing ended up being the interesting part, not the final score.

## What is here

- `eda.py` reads the ten genre CSVs and writes the plots in `images/`, with distributions, box plots and violins per genre, plus a correlation heatmap.
- `preprocessing.py` builds the scikit-learn pipeline: missing value handling, an artist to genre encoder, one hot encoding for categoricals, scaling, `SelectKBest` and PCA. It writes the train and validation splits to `data/`.
- `trainer.py` trains a stacking classifier, with random forest, gradient boosting and XGBoost under a logistic regression meta model, then writes predictions.
- `results.csv` is the 12,435 row prediction output from the October 2024 run. `combined_data.csv` is the merged dataset.

## Running it

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
python eda.py
python preprocessing.py
python trainer.py
```

`training-data/` holds the ten genre files. The test split the pipeline expects at `testing-data/testing-instances.csv` is not in the repo, so `preprocessing.py` needs a test file passed in and `trainer.py` needs processed test rows before it will predict. Point both at your own split, or hold out a slice of `training-data/`.

## State

Written for a September 2024 assignment and left close to how it was submitted. The data outputs stay in the repo so the plots can be read without rerunning anything.

MIT licensed.
