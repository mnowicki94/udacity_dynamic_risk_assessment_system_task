"""
Author: Maciej Nowicki
Date: 2023-10-05
Description: This script contains functions for diagnostics of a machine learning model, including model predictions, summary statistics, missing data checks, execution time measurements, and outdated package checks.
"""

import pandas as pd
import numpy as np
import timeit
import os
import json
import logging
from dir_conf import prod_deployment_path, test_data_path
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/diagnostics.log", mode="w"),
        logging.StreamHandler(),
    ],
)


##################Function to get model predictions
def model_predictions(prod_deployment_path, test_data_path, filename):
    # read the deployed model and a test dataset, calculate predictions

    model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Load the test data
    test_data = pd.read_csv(os.path.join(test_data_path, filename))
    X_test = test_data.drop(columns=["exited", "corporation"])
    y_test = test_data["exited"]

    # Make predictions
    y_pred = model.predict(X_test)

    logging.info(f"Made predictions on test data: {y_pred}")
    return y_pred


##################Function to get summary statistics
def dataframe_summary(test_data_path, filename):
    # calculate summary statistics here
    df = pd.read_csv(os.path.join(test_data_path, filename))
    df = df.drop(columns=["corporation"])

    summary_stats = df.describe().to_dict()
    logging.info(f"Summary statistics: {summary_stats}")
    return summary_stats


##################Function to check for missing data
def missing_data(test_data_path, filename):
    # read the dataset
    df = pd.read_csv(os.path.join(test_data_path, filename))

    # count the number of NA values in each column
    na_counts = df.isna().sum()

    # calculate the percentage of NA values in each column
    na_percentage = (na_counts / len(df)) * 100

    na_percentage_dict = na_percentage.to_dict()
    logging.info(f"Missing data percentages: {na_percentage_dict}")
    return na_percentage.tolist()


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    timing = {}
    for script in ["src/training.py", "src/ingestion.py"]:
        start_time = timeit.default_timer()
        os.system(f"python {script}")
        timing[script] = timeit.default_timer() - start_time

    logging.info(f"Execution time: {timing}")
    return timing


##################Function to check dependencies
def outdated_packages_list():
    # get a list of outdated packages
    result = os.popen("pip list --outdated --format=json").read()
    outdated_packages = json.loads(result)

    outdated_list = []
    for package in outdated_packages:
        outdated_list.append(
            {
                "name": package["name"],
                "current_version": package["version"],
                "latest_version": package["latest_version"],
            }
        )

    logging.info(f"Outdated packages: {outdated_list}")
    return outdated_list


if __name__ == "__main__":
    filename = "testdata.csv"

    model_predictions(prod_deployment_path, test_data_path, filename)
    dataframe_summary(test_data_path, filename)
    missing_data(test_data_path, filename)
    execution_time()
    outdated_packages_list()
