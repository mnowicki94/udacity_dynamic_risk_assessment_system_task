"""
Author: Maciej Nowicki
Date: 2023-10-05
Description: This script calls multiple API endpoints to retrieve predictions, scoring, summary statistics,
and diagnostics for a given dataset. The responses from these endpoints are then combined and written to a file.
"""

import os
import requests
import json
from dir_conf import output_model_path

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
response1 = requests.get(f"{URL}/prediction?filename=testdata.csv")
response2 = requests.get(f"{URL}/scoring?filename=testdata.csv")
response3 = requests.get(f"{URL}/summarystats?filename=testdata.csv")
response4 = requests.get(f"{URL}/diagnostics?filename=testdata.csv")

# Combine all API responses
responses = {
    "prediction": response1.json(),
    "scoring": response2.json(),
    "summarystats": response3.json(),
    "diagnostics": response4.json(),
}

# Write the responses to your workspace
with open(
    os.path.join(output_model_path, "apireturns.txt"),
    "w",
) as outfile:
    json.dump(responses, outfile, indent=4)
