import os
import pandas as pd
from pathlib import Path

from .enums import DatasetCSV


def import_data(correction: bool = True):
    """Import the data from the confident_data.csv file if it exists, otherwise it's computed from the raw data."""
    file = "corrected_confident_data.csv" if correction else "confident_data.csv"

    if "confident_data.csv" in os.listdir(str(Path(DatasetCSV.CLEAN.value).parent)):
        return pd.read_csv(Path(DatasetCSV.CLEAN.value).parent / file)
    else:
        raise ValueError("The confident_data.csv file does not exist. You must run the correction first.")
