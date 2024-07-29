"""
This script will load all the csv files in the data folder with pandas and checks the dimension of the pandas dataframe.
It is expected to have only two columns per csv file and at least 1 row.
"""

import os

import pandas as pd
import pytest

from spartacus import DataFolder, Spartacus, DatasetCSV, RowData


@pytest.mark.parametrize("data_folder", DataFolder)
def test_data_format(data_folder):

    for subfile in os.listdir(data_folder.value):
        if subfile.endswith(".csv") and not subfile.startswith("corrections"):
            print("Loading file:", subfile)
            df = pd.read_csv(os.path.join(data_folder.value, subfile), header=None)
            print("Shape:", df.shape)
            print("Columns:", df.columns)
            print(df.head())
            df.columns = ["humerothoracic_angle", "value"]
            print("")

            if df.shape[1] != 2:
                raise ValueError("The csv file should have only two columns.")
            if df.shape[0] < 1:
                raise ValueError("The csv file should have at least one row.")

            # verify that the first column is a float
            try:
                df["humerothoracic_angle"] = df["humerothoracic_angle"].astype(float)
            except ValueError:
                raise ValueError("The first column should be a float.")

            # verify that the second column is a float
            try:
                df["value"] = df["value"].astype(float)
            except ValueError:
                raise ValueError("The second column should be a float.")

    print("All csv files have been loaded successfully.")


def test_data_loading():
    # open the file only_dataset_raw.csv
    df = pd.read_csv(DatasetCSV.CLEAN.value)
    df = df[df["dataset_authors"] == "Guttierrez Delgado et al."]
    print(df.shape)
    sp = Spartacus(dataframe=df)
    for i, row in sp.dataframe.iterrows():
        row_data = RowData(row)
        row_data.import_data()

    print(df.shape)
    return sp
