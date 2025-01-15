from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import logging
from dir_conf import output_model_path, test_data_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/scoring.log"), logging.StreamHandler()],
)


#################Function for model scoring
def score_model(output_model_path, test_data_path):
    # Load the trained model
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    X_test = test_data.drop(columns=["exited", "corporation"])
    y_test = test_data["exited"]

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the F1 score
    f1_score = metrics.f1_score(y_test, y_pred)

    # Write the F1 score to the latestscore.txt file
    score_path = os.path.join(output_model_path, "latestscore.txt")
    with open(score_path, "w") as score_file:
        score_file.write(f"F1 Score: {f1_score}\n")

    logging.info(f"Model F1 Score: {f1_score}")


if __name__ == "__main__":
    score_model(output_model_path, test_data_path)
