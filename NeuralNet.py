#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import argparse
import itertools
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# Public UCI path for the Breast Cancer Wisconsin (Diagnostic) dataset
DEFAULT_DATA_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                    "breast-cancer-wisconsin/wdbc.data")

# wdbc.data has no header row: ID, Diagnosis, then mean / SE / worst of 10 features
BASE_FEATURES = ['radius', 'texture', 'perimeter', 'area', 'smoothness',
                 'compactness', 'concavity', 'concave_points', 'symmetry',
                 'fractal_dimension']
COLUMN_NAMES = ['ID', 'Diagnosis'] + [f"{f}_{stat}"
                                      for stat in ('mean', 'se', 'worst')
                                      for f in BASE_FEATURES]

# Plot colors (one per learning-rate x hidden-layer combination) and ink
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK_SECONDARY = '#52514e'
GRID_COLOR = '#e1e0d9'


class NeuralNet:
    def __init__(self, dataFile, header=True, seed=42, test_size=0.2,
                 neurons=16, out_dir="results"):
        self.seed = seed
        self.test_size = test_size
        self.neurons = neurons
        self.out_dir = out_dir
        self.raw_input = pd.read_csv(dataFile, header=0 if header else None)
        if not header:
            if self.raw_input.shape[1] != len(COLUMN_NAMES):
                raise ValueError(f"Expected {len(COLUMN_NAMES)} columns, "
                                 f"got {self.raw_input.shape[1]}")
            self.raw_input.columns = COLUMN_NAMES

        # Sanity check of the loaded data
        print(f"Loaded {dataFile}")
        print(f"Shape: {self.raw_input.shape}")
        print(f"Class counts:\n{self.raw_input['Diagnosis'].value_counts().to_string()}")
        print(f"Null values: {int(self.raw_input.isnull().sum().sum())}")




    # Pre-processing: drop ID, integrity checks, null handling, label encoding,
    # train/test split and standardization. The data is split before
    # standardizing so the scaler is fit on training data only (no test-set
    # leakage).
    def preprocess(self):
        data = self.raw_input.copy()

        # ID is just a sample identifier, not a predictive feature
        data = data.drop(columns=['ID'])

        # Integrity: treat '?' / blank strings as missing, force features to numeric
        data = data.replace(['?', ''], np.nan)
        feature_cols = [c for c in data.columns if c != 'Diagnosis']
        data[feature_cols] = data[feature_cols].apply(pd.to_numeric, errors='coerce')

        # Integrity: remove duplicate rows and rows with missing/invalid labels
        n = len(data)
        data = data.drop_duplicates()
        print(f"Duplicate rows removed: {n - len(data)}")

        n = len(data)
        data['Diagnosis'] = data['Diagnosis'].astype(str).str.strip().str.upper()
        data = data[data['Diagnosis'].isin(['M', 'B'])]
        print(f"Rows with missing/invalid label removed: {n - len(data)}")

        # Nulls in features: impute with the column median
        n_null = int(data[feature_cols].isnull().sum().sum())
        data[feature_cols] = data[feature_cols].fillna(data[feature_cols].median())
        print(f"Missing feature values imputed with median: {n_null}")

        # Categorical to numerical: M (malignant) = 1, B (benign) = 0
        data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

        # Put the label last, as train_evaluate expects
        data = data[feature_cols + ['Diagnosis']].reset_index(drop=True)

        self.processed_data = data
        print(f"Processed data shape: {data.shape}")

        # Stratified, seeded train/test split
        ncols = len(data.columns)
        X = data.iloc[:, 0:(ncols - 1)]
        y = data.iloc[:, (ncols - 1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.seed)
        print(f"Train size: {len(X_train)}, test size: {len(X_test)}")

        # Standardize with statistics from the training set only
        scaler = StandardScaler().fit(X_train)
        self.X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train.to_numpy()
        self.y_test = y_test.to_numpy()
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        activations = ['logistic', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]

        # Create the neural network and be sure to keep track of the performance
        #   metrics

        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0

    # Train one network one epoch at a time and record its history.
    # Plain SGD (no momentum) so the learning rate is the step size eta.
    def train_model(self, activation, lr, epochs, n_layers, data):
        X_train, y_train, X_test, y_test = data
        clf = MLPClassifier(hidden_layer_sizes=(self.neurons,) * n_layers,
                            activation=activation, solver='sgd',
                            learning_rate_init=lr, momentum=0.0,
                            random_state=self.seed)

        history = {k: [] for k in ('train_acc', 'test_acc', 'train_mse',
                                   'test_mse', 'train_logloss', 'test_logloss')}
        for _ in range(epochs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.partial_fit(X_train, y_train, classes=[0, 1])
            for split, X_s, y_s in (('train', X_train, y_train),
                                    ('test', X_test, y_test)):
                proba = clf.predict_proba(X_s)[:, 1]
                history[f'{split}_acc'].append(
                    accuracy_score(y_s, (proba >= 0.5).astype(int)))
                history[f'{split}_mse'].append(mean_squared_error(y_s, proba))
                history[f'{split}_logloss'].append(
                    log_loss(y_s, proba, labels=[0, 1]))

        print(f"Trained act={activation:<8} lr={lr:<4} epochs={epochs:<3} "
              f"layers={n_layers}: test acc {history['test_acc'][-1]:.4f}")
        return history

    # One figure per metric: rows are train / test, columns are activations.
    # Within a panel, color = (learning rate, hidden layers) and line style =
    # epochs, so each of the 24 models has a unique color + style.
    def plots(self, activations, learning_rates, epochs_list,
                        layers_list):
        combos = list(itertools.product(learning_rates, layers_list))
        color_of = {c: SERIES_COLORS[i] for i, c in enumerate(combos)}
        style_of = {epochs_list[0]: (0, (4, 2)), epochs_list[-1]: '-'}

        for metric, ylabel in (('acc', 'Accuracy'), ('mse', 'MSE')):
            fig, axes = plt.subplots(2, len(activations), figsize=(16, 9),
                                     sharex=True, sharey=True)
            for row, split in enumerate(('train', 'test')):
                for col, act in enumerate(activations):
                    ax = axes[row, col]
                    # Draw longer runs first so shorter runs sit on top
                    for epochs in sorted(epochs_list, reverse=True):
                        for lr, n_layers in combos:
                            values = self.histories[(act, lr, epochs, n_layers)][f'{split}_{metric}']
                            ax.plot(range(1, len(values) + 1), values,
                                    color=color_of[(lr, n_layers)],
                                    linestyle=style_of[epochs], linewidth=2,
                                    label=f"lr={lr}, layers={n_layers}, epochs={epochs}")
                            ax.plot(len(values), values[-1], marker='o',
                                    markersize=5, color=color_of[(lr, n_layers)])
                    ax.set_title(f"{act} ({split})", fontsize=12)
                    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
                    ax.tick_params(colors=INK_SECONDARY)
                    for spine in ('top', 'right'):
                        ax.spines[spine].set_visible(False)
                    if col == 0:
                        ax.set_ylabel(f"{split.capitalize()} {ylabel}")
                    if row == 1:
                        ax.set_xlabel("Epoch")

            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', ncol=4,
                       frameon=False, fontsize=10)
            fig.suptitle(f"Model history: {ylabel} vs epoch for all "
                         f"{len(self.histories)} models", fontsize=14)
            fig.tight_layout(rect=(0, 0.08, 1, 0.96))
            path = os.path.join(self.out_dir, f"history_{metric}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Saved {path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network hyperparameter analysis")
    parser.add_argument("--data", default=DEFAULT_DATA_URL,
                        help="Path or URL to wdbc.data (default: public UCI URL)")
    args = parser.parse_args()

    neural_network = NeuralNet(args.data, header=False)
    neural_network.preprocess()
    # TODO: enable once train_evaluate is implemented
    # neural_network.train_evaluate()
