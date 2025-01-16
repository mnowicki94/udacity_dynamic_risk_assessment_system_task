"""
Author: Maciej Nowicki
Date: 2023-10-20
Description: This script checks for new data, trains a model, checks for model drift, and redeploys the model if necessary.
"""

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import dir_conf
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib
import pandas as pd
import pickle

##################Check and read new data
# first, read ingestedfiles.txt


def check_for_new_files():
    # check what files was previously ingested
    with open(
        os.path.join(dir_conf.prod_deployment_path, "ingestedfiles.txt"), "r"
    ) as f:
        ingested_files = f.read().splitlines()

    print(f"Prevoisuly ingested files: {ingested_files}")

    # check files in prod data folder

    prod_files = [
        f for f in os.listdir(dir_conf.input_folder_path) if f.endswith(".csv")
    ]
    print(f"Files in prod data folder: {prod_files}")

    # check if there are new files
    new_files = set(prod_files) - set(ingested_files)
    print(f"New files found: {new_files}")
    if len(new_files) > 0:
        new_files_present = True
        print(f"New files found: {new_files_present}")

    return new_files_present


def run_training_process():
    print("New files found. Running the ingestion, training, and scoring steps.")

    # Ingest new data
    dataframes = []
    for file in os.listdir(dir_conf.input_folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(dir_conf.input_folder_path, file))
            dataframes.append(df)
    combined_df = pd.concat(dataframes)
    combined_df = combined_df.drop_duplicates()

    # Train model
    X = df.drop(columns=["exited", "corporation"])
    y = df["exited"]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Import the model

    # import the model
    model_path = os.path.join(dir_conf.prod_deployment_path, "trainedmodel.pkl")
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Fit the logistic regression to your data
    model.fit(X_train, y_train)

    # Score model
    y_pred = model.predict(X_test)
    new_score = f1_score(y_test, y_pred)

    print(f"New model score: {new_score}")
    return new_score


def check_model_drift(new_score):
    # Read the latest score from the deployed model
    with open(os.path.join(dir_conf.prod_deployment_path, "latestscore.txt"), "r") as f:
        latest_score = float(f.read().split(":")[1])

    print(f"Latest score: {latest_score}")

    # Check for model drift
    if latest_score > new_score:
        model_drift = True
        print("Model drift detected.")
    else:
        model_drift = False
        print("No model drift detected.")

    return model_drift


##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
if __name__ == "__main__":
    filename = "testdata.csv"
    new_files_present = check_for_new_files()
    if new_files_present:
        new_score = run_training_process()
        model_drift = check_model_drift(new_score)
        if model_drift:
            ingestion.merge_multiple_dataframe(
                dir_conf.input_folder_path, dir_conf.output_folder_path
            )
            training.train_model(
                dir_conf.output_folder_path, dir_conf.output_model_path
            )
            scoring.score_model(
                dir_conf.output_model_path, dir_conf.test_data_path, filename
            )
            deployment.deploy_model(
                dir_conf.output_model_path, dir_conf.prod_deployment_path
            )
            diagnostics.model_predictions(
                dir_conf.prod_deployment_path, dir_conf.test_data_path, filename
            )
            diagnostics.dataframe_summary(dir_conf.test_data_path, filename)
            diagnostics.missing_data(dir_conf.test_data_path, filename)
            diagnostics.execution_time()
            diagnostics.outdated_packages_list()
            reporting.reporting(
                dir_conf.test_data_path,
                dir_conf.prod_deployment_path,
                dir_conf.output_model_path,
            )
            os.system("python src/apicalls.py")
        else:
            print("No model drift detected. Ending the process here.")
    else:
        print("No new files found. Ending the process here.")
