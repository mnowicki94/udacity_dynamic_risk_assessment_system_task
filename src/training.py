"""
Author: Maciej Nowicki
Date: 2023-10-05

This script trains a logistic regression model using the data from the specified output folder path.
The trained model is saved to the specified output model path.

Functions:
    train_model(output_folder_path, output_model_path): Trains the logistic regression model and saves it.

Usage:
    Run this script to train the model and save it to the specified path.
"""

import logging
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from dir_conf import output_folder_path, output_model_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/training.log"), logging.StreamHandler()],
)


#################Function for training the model
def train_model(output_folder_path, output_model_path):
    """
    Trains a logistic regression model using the data from the specified output folder path.
    The trained model is saved to the specified output model path.

    Args:
        output_folder_path (str): The path to the folder containing the training data.
        output_model_path (str): The path to the folder where the trained model will be saved.
    """
    logging.info("Starting model training.")

    # use this logistic regression for training
    filename = "finaldata.csv"

    df = pd.read_csv(os.path.join(output_folder_path, filename))
    logging.info(f"Data loaded from {filename}.")

    # Split the data into features and target
    X = df.drop(columns=["exited", "corporation"])
    y = df["exited"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    logging.info("Data split into training and test sets.")

    # Initialize the logistic regression model
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",  # Updated parameter
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # Fit the logistic regression to your data
    model.fit(X_train, y_train)
    logging.info("Model training completed.")

    # Write the trained model to your workspace in a file called trainedmodel.pkl
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(model, model_file)
    logging.info(f"Model saved to {model_path}")

    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    train_model(output_folder_path, output_model_path)
