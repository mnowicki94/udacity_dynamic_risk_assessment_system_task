"""
Author: Maciej Nowicki
Date: 2023-10-05
Description: This script calculates a confusion matrix using test data and a deployed model, and saves the confusion matrix as an image file.
"""

import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from dir_conf import test_data_path, prod_deployment_path, output_model_path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/reporting.log", mode="w"),
        logging.StreamHandler(),
    ],
)


def reporting(test_data_path, prod_deployment_path, output_model_path):
    """
    Calculate a confusion matrix using the test data and the deployed model.
    Save the confusion matrix as an image file.

    Parameters:
    test_data_path (str): Path to the test data directory.
    prod_deployment_path (str): Path to the production deployment directory.
    output_model_path (str): Path to the output model directory.

    Returns:
    y_pred (array): Predicted values.
    """
    model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X_test = test_data.drop(columns=["exited", "corporation"])
    y_test = test_data["exited"]

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    logging.info(f"Confusion Matrix:\n{cm}")

    # Save confusion matrix to file
    cm_path = os.path.join(output_model_path, "confusion_matrix.png")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(cm_path)
    logging.info(f"Confusion matrix saved to {cm_path}")

    return y_pred


if __name__ == "__main__":
    reporting(test_data_path, prod_deployment_path, output_model_path)
