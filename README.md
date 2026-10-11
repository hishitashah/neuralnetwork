# Assignment 2: Neural Networks (Programming Part)

**Group members:** TODO Name (netid), TODO Name (netid)

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The code reads the data straight from the public UCI file, so no data is included in this submission:
`https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data`

- 569 instances, 30 real-valued features, binary label `Diagnosis` (M = malignant, B = benign)
- The file has no header row. The code assigns the column names: `ID`, `Diagnosis`, and `<feature>_{mean,se,worst}` for the 10 base features.

## Setup
Requires Python 3.8+.
```
pip install -r requirements.txt
```

## How to run
```
# uses the public UCI URL by default
python NeuralNet.py

# or point to any local path / URL
python NeuralNet.py --data path/to/wdbc.data
```

## Modifications to the template
- `__main__` reads the data path from the `--data` command-line argument instead of a hardcoded `"train.csv"`.
- `__init__` now uses the `header` flag (`header=False` for wdbc.data) and assigns the column names.
- `__init__` prints a quick summary of the loaded data: shape, class counts, and null count.
- `preprocess` drops `ID`, coerces features to numeric (`?`/blank become NaN), removes duplicates and rows with an invalid label, imputes missing features with the column median, encodes `Diagnosis` as M = 1 / B = 0, and moves it to the last column.
- Standardization is not done in `preprocess`. It will happen in `train_evaluate`, after the train/test split, with the scaler fit on the training set only to avoid leaking test data.

## Results
TODO

## Report
TODO

## Assumptions
TODO

## References
- UCI ML Repository, Breast Cancer Wisconsin (Diagnostic) dataset
- scikit-learn documentation: `MLPClassifier`
