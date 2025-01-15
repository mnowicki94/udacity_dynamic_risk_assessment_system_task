import os
import json

project_name = "udacity_dynamic_risk_assessment_system_task"

# Load configuration from config.json and set path variables
config_path = os.path.join(os.path.dirname(__file__), "../config.json")
with open(config_path, "r") as config_file:
    config = json.load(config_file)

input_folder_path = os.path.join(
    os.path.abspath(f"../{project_name}"), "data", config["input_folder_path"]
)
output_folder_path = os.path.join(
    os.path.abspath(f"../{project_name}"), "data", config["output_folder_path"]
)
test_data_path = os.path.join(
    os.path.abspath(f"../{project_name}"), "data", config["test_data_path"]
)
output_model_path = os.path.join(
    os.path.abspath(f"../{project_name}"), "models", config["output_model_path"]
)
prod_deployment_path = os.path.join(
    os.path.abspath(f"../{project_name}"), "models", config["prod_deployment_path"]
)
