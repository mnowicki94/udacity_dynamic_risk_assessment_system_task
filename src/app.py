"""
Author: Maciej Nowicki
Date: 2023-10-05
Description: This script sets up a Flask web application with multiple endpoints for
             prediction, scoring, summary statistics, and diagnostics for a dynamic
             risk assessment system.
"""

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics
import scoring
import reporting
import json
import os
from dir_conf import test_data_path, prod_deployment_path, output_model_path


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"


@app.route("/")
def index():
    return "Hi World"


#######################Prediction Endpoint
@app.route("/prediction", methods=["GET", "OPTIONS"])
def predict():
    filename = request.args.get("filename")

    # Call the prediction function
    predictions = diagnostics.model_predictions(
        prod_deployment_path, test_data_path, filename
    )
    # Return predictions as JSON
    return jsonify(predictions.tolist())


# #######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    filename = request.args.get("filename")
    score = scoring.score_model(output_model_path, test_data_path, filename)
    # Return score as JSON
    # http://127.0.0.1:8000/scoring?filename=testdata.csv
    return jsonify(score.item())


# #######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    filename = request.args.get("filename")
    stats = diagnostics.dataframe_summary(test_data_path, filename)
    # Return stats as JSON
    return jsonify(stats)


# #######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def other_diagnostics():
    filename = request.args.get("filename")
    time = diagnostics.execution_time()
    nas = diagnostics.missing_data(test_data_path, filename)
    outdated = diagnostics.outdated_packages_list()

    return jsonify({"time": time, "missing_data": nas, "outdated_packages": outdated})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=True)
