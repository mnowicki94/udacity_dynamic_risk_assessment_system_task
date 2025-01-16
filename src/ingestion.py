"""
Author: Maciej Nowicki
Date: 2023-10-05
Description: This script ingests multiple CSV files from a specified input directory,
             merges them into a single DataFrame, removes duplicates, and writes the
             combined DataFrame to an output CSV file. It also logs the names of the
             ingested files.
"""

import pandas as pd
import os
import logging
from dir_conf import input_folder_path, output_folder_path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/ingestion.log", mode="w"),
        logging.StreamHandler(),
    ],
)

# Create a logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")


############# Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    """
    Merge multiple CSV files from the input folder into a single DataFrame, remove duplicates,
    and save the result to a CSV file in the output folder. Also, log the names of the ingested files.

    Args:
        input_folder_path (str): Path to the folder containing input CSV files.
        output_folder_path (str): Path to the folder where the output CSV file and log file will be saved.
    """
    logging.info("Starting data ingestion process.")

    # Check for datasets, compile them together, and write to an output file
    filenames = [f for f in os.listdir(input_folder_path) if f.endswith(".csv")]
    dataframes = []

    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as f:
        for filename in filenames:
            df = pd.read_csv(os.path.join(input_folder_path, filename))
            dataframes.append(df)
            f.write(f"{filename}\n")
            logging.info(f"Ingested file: {filename}")

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    combined_df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)

    logging.info("Data ingestion process completed successfully.")
    logging.info(
        f"Combined data saved to {os.path.join(output_folder_path, 'finaldata.csv')}"
    )


if __name__ == "__main__":
    merge_multiple_dataframe(input_folder_path, output_folder_path)
