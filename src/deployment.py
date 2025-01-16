"""
Author: Maciej Nowicki
Date: October 2023
Description: This script handles the deployment of the trained model by copying necessary files to the deployment directory.
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from dir_conf import output_model_path, prod_deployment_path, output_folder_path
import shutil
import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/deployment.log", mode="w"),
        logging.StreamHandler(),
    ],
)


def deploy_model(output_model_path, prod_deployment_path):
    """
    Copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory.

    Parameters:
    output_model_path (str): Path to the directory containing the output model files.
    prod_deployment_path (str): Path to the deployment directory where files will be copied.
    """
    # Copy the pickle file to the deployment directory
    for file in ["trainedmodel.pkl", "latestscore.txt"]:
        logging.info(f"Processing file: {file}")
        model_path = os.path.join(output_model_path, file)
        deployment_model_path = os.path.join(prod_deployment_path, file)
        if os.path.exists(model_path):
            shutil.copy(model_path, deployment_model_path)
            logging.info(f"Copied {file} to deployment directory")
        else:
            logging.warning(f"{file} does not exist in the output model path")

    file = "ingestedfiles.txt"
    logging.info(f"Processing file: {file}")
    model_path = os.path.join(output_folder_path, file)
    deployment_model_path = os.path.join(prod_deployment_path, file)
    if os.path.exists(model_path):
        shutil.copy(model_path, deployment_model_path)
        logging.info(f"Copied {file} to deployment directory")
    else:
        logging.warning(f"{file} does not exist in the output model path")


if __name__ == "__main__":
    deploy_model(output_model_path, prod_deployment_path)
