# Neural Network Project

**Group members:** Hishita Shah (hrs220004), Megan Chacko (msc220005)

## Dataset
Breast Cancer Wisconsin (Diagnostic), UCI ML Repository:
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The code reads the data straight from the public UCI file, so no data is included in this submission:
`https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data`

- 569 instances, 30 real-valued features, binary label `Diagnosis` (M = malignant, B = benign)
- The file has no header row. The code assigns the column names: `ID`, `Diagnosis`, and `<feature>_{mean,se,worst}` for the 10 base features.

## Setup
Requires Python 3.9 or newer and an internet connection (the dataset is downloaded from the UCI URL at run time).

Install the following packages before running:

| Package | Used for |
|---|---|
| `numpy` | numerical arrays |
| `pandas` | loading and cleaning the data, results table |
| `scikit-learn` | train/test split, `StandardScaler`, `MLPClassifier`, metrics |
| `matplotlib` | model history plots |
| `tabulate` | saving the results table as markdown (`results.md`) |

```
pip install numpy pandas scikit-learn matplotlib tabulate
```

Tested with Python 3.14.7, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, matplotlib 3.11.2 and tabulate 0.10.0.

## How to run
```
# uses the public UCI URL by default
python NeuralNet.py

# or point to any local path / URL
python NeuralNet.py --data path/to/wdbc.data
```

The run prints the data-cleaning counts (duplicate rows removed, rows with an invalid label removed, missing values filled), the train/test sizes, one line per trained model, the full results table, the best model, and the mean test accuracy for each hyperparameter value. 

## Changes to template
- `__main__` reads the data path from the `--data` command-line argument instead of a hardcoded `"train.csv"`.
- `__init__` now uses the `header` flag (`header=False` for wdbc.data) and assigns the column names.
- `preprocess` drops `ID`, coerces features to numeric (`?`/blank become NaN), removes duplicates and rows with an invalid label, imputes missing features with the column median, encodes `Diagnosis` as M = 1 / B = 0, and moves it to the last column.
- `preprocess` also does the train/test split and then standardizes the features (`StandardScaler`). The split happens first so the scaler is fit on the training set only and no test data leaks into training. It stores the standardized `X_train`, `X_test`, `y_train`, `y_test`, which `train_evaluate` uses directly instead of splitting the data itself.
- `__init__` takes optional `seed` (42), `test_size` (0.2), `neurons` (16) and `out_dir` (`"results"`) arguments.
- The split is stratified and seeded (`train_test_split(..., stratify=y, random_state=seed)`) instead of the template's unseeded default, so runs are reproducible.
- `train_evaluate` keeps the template's hyperparameter grid and trains every combination (3 × 2 × 2 × 2 = 24 models). Two helper methods were added:
  - `train_model` trains one `MLPClassifier` one epoch at a time with `partial_fit` and records train/test accuracy, MSE and log-loss after each epoch.
  - `plots` draws the model-history plots.
- Errors are reported as MSE and as log-loss.

## Results
All 24 models, sorted by test accuracy (ties broken by test MSE). Train/test sizes are 455/114.

| Activation   |   Learning Rate |   Epochs |   Hidden Layers |   Neurons/Layer |   Train Acc |   Test Acc |   Train MSE |   Test MSE |   Train LogLoss |   Test LogLoss |
|:-------------|----------------:|---------:|----------------:|----------------:|------------:|-----------:|------------:|-----------:|----------------:|---------------:|
| relu         |          0.1000 |      200 |               3 |              16 |      0.9934 |     0.9825 |      0.0068 |     0.0160 |          0.0257 |         0.0877 |
| relu         |          0.1000 |      100 |               2 |              16 |      0.9890 |     0.9825 |      0.0117 |     0.0174 |          0.0471 |         0.0746 |
| relu         |          0.1000 |      100 |               3 |              16 |      0.9824 |     0.9825 |      0.0127 |     0.0179 |          0.0467 |         0.0827 |
| tanh         |          0.0100 |      200 |               3 |              16 |      0.9780 |     0.9825 |      0.0228 |     0.0183 |          0.0947 |         0.0832 |
| relu         |          0.1000 |      200 |               2 |              16 |      0.9956 |     0.9825 |      0.0070 |     0.0193 |          0.0290 |         0.0920 |
| logistic     |          0.1000 |      200 |               2 |              16 |      0.9736 |     0.9825 |      0.0224 |     0.0208 |          0.0965 |         0.0905 |
| tanh         |          0.1000 |      100 |               2 |              16 |      0.9824 |     0.9737 |      0.0139 |     0.0174 |          0.0561 |         0.0691 |
| tanh         |          0.1000 |      100 |               3 |              16 |      0.9824 |     0.9737 |      0.0113 |     0.0176 |          0.0469 |         0.0745 |
| tanh         |          0.1000 |      200 |               3 |              16 |      0.9934 |     0.9737 |      0.0061 |     0.0212 |          0.0278 |         0.0897 |
| tanh         |          0.1000 |      200 |               2 |              16 |      0.9912 |     0.9737 |      0.0099 |     0.0229 |          0.0410 |         0.0871 |
| tanh         |          0.0100 |      200 |               2 |              16 |      0.9692 |     0.9737 |      0.0270 |     0.0234 |          0.1094 |         0.0975 |
| relu         |          0.0100 |      200 |               3 |              16 |      0.9429 |     0.9649 |      0.0361 |     0.0276 |          0.1252 |         0.1032 |
| relu         |          0.0100 |      200 |               2 |              16 |      0.9516 |     0.9649 |      0.0333 |     0.0313 |          0.1221 |         0.1161 |
| tanh         |          0.0100 |      100 |               3 |              16 |      0.9451 |     0.9561 |      0.0404 |     0.0319 |          0.1571 |         0.1354 |
| tanh         |          0.0100 |      100 |               2 |              16 |      0.9473 |     0.9474 |      0.0409 |     0.0367 |          0.1620 |         0.1496 |
| relu         |          0.0100 |      100 |               3 |              16 |      0.9297 |     0.9386 |      0.0593 |     0.0499 |          0.2087 |         0.1863 |
| logistic     |          0.1000 |      100 |               2 |              16 |      0.9363 |     0.9386 |      0.0742 |     0.0726 |          0.2921 |         0.2877 |
| relu         |          0.0100 |      100 |               2 |              16 |      0.9385 |     0.9298 |      0.0526 |     0.0512 |          0.1942 |         0.1912 |
| logistic     |          0.0100 |      200 |               2 |              16 |      0.6264 |     0.6316 |      0.2280 |     0.2269 |          0.6481 |         0.6459 |
| logistic     |          0.1000 |      200 |               3 |              16 |      0.6264 |     0.6316 |      0.2292 |     0.2281 |          0.6507 |         0.6484 |
| logistic     |          0.0100 |      100 |               2 |              16 |      0.6264 |     0.6316 |      0.2314 |     0.2301 |          0.6552 |         0.6527 |
| logistic     |          0.1000 |      100 |               3 |              16 |      0.6264 |     0.6316 |      0.2330 |     0.2317 |          0.6586 |         0.6561 |
| logistic     |          0.0100 |      200 |               3 |              16 |      0.6264 |     0.6316 |      0.2339 |     0.2326 |          0.6605 |         0.6579 |
| logistic     |          0.0100 |      100 |               3 |              16 |      0.6264 |     0.6316 |      0.2341 |     0.2329 |          0.6610 |         0.6586 |

Mean test accuracy for each hyperparameter value:

| Hyperparameter | Value → mean test accuracy |
|---|---|
| Activation | logistic 0.714 · tanh 0.969 · relu 0.966 |
| Learning rate | 0.01 → 0.849 · 0.1 → 0.917 |
| Epochs | 100 → 0.877 · 200 → 0.890 |
| Hidden layers | 2 → 0.909 · 3 → 0.857 |

For MSE vs epoch (100-epoch), each 100-epoch model uses the same seed as its 200-epoch counterpart, so its curve is identical for the first 100 epochs and lies underneath the solid line. The dot at epoch 100 marks where the 100-epoch model stopped.

## References
- UCI ML Repository, Breast Cancer Wisconsin (Diagnostic) dataset
- scikit-learn documentation: `MLPClassifier`
