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

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


class NeuralNet:
    def __init__(self, dataFile, header=True):
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




    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input

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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural network hyperparameter analysis")
    parser.add_argument("--data", default=DEFAULT_DATA_URL,
                        help="Path or URL to wdbc.data (default: public UCI URL)")
    args = parser.parse_args()

    neural_network = NeuralNet(args.data, header=False)
    # TODO: enable once preprocess/train_evaluate are implemented
    # neural_network.preprocess()
    # neural_network.train_evaluate()
