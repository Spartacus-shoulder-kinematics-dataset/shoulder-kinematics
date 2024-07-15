"""
This module is used to load the data from the csv file for individual datasets for each dofs.
"""

import numpy as np
import pandas as pd


def load_euler_csv(csv_filenames: tuple[str, str, str], drop_humerothoracic_raw_data: bool = True) -> pd.DataFrame:
    """
    Load the csv file from the filename and return a pandas dataframe.
    """
    df = pd.DataFrame(columns=["humerothoracic_angle"])

    nb_files = len([x for x in csv_filenames if x is not None])
    dof_idx = [i for (i, _) in enumerate(csv_filenames) if _ is not None]

    # Initialize an empty list to store the loaded dataframes
    csv_files_dof = []

    # Loop over the filenames and load each file
    for i, csv_filename in enumerate(csv_filenames):
        if csv_filename is None:
            csv_files_dof.append(None)
            continue

        csv_file = load_csv(
            csv_filename,
            [
                f"humerothoracic_angle_dof{i + 1}",
                f"value_dof{i + 1}",
            ],
        )
        csv_files_dof.append(csv_file)

    concatenated_dataframe = pd.concat([df] + csv_files_dof, axis=1)

    if nb_files > 1 and not all(
        concatenated_dataframe[f"humerothoracic_angle_dof{i + 1}"].equals(
            concatenated_dataframe["humerothoracic_angle_dof1"]
        )
        for i in dof_idx[1:]
    ):

        print("The dofs column abscissas are not the same: Interpolating through the minimal range")
        # Interpolating through the minimal range
        min_value = max(concatenated_dataframe[f"humerothoracic_angle_dof{i + 1}"].min() for i in dof_idx)
        max_value = min(concatenated_dataframe[f"humerothoracic_angle_dof{i + 1}"].max() for i in dof_idx)
        number_of_points = min(len(concatenated_dataframe[f"humerothoracic_angle_dof{i+1}"]) for i in dof_idx)

        interpolated_range = np.linspace(min_value, max_value, number_of_points)

        interpolated_values = [
            np.interp(
                interpolated_range,
                concatenated_dataframe[f"humerothoracic_angle_dof{i+1}"],
                concatenated_dataframe[f"value_dof{i+1}"],
            )
            for i in dof_idx
        ]

        # replace the values
        concatenated_dataframe = pd.DataFrame(
            columns=["humerothoracic_angle", "value_dof1", "value_dof2", "value_dof3"]
        )
        concatenated_dataframe["humerothoracic_angle"] = interpolated_range
        for i, interpolated_value in zip(dof_idx, interpolated_values):
            concatenated_dataframe[f"value_dof{i + 1}"] = interpolated_value

    else:
        concatenated_dataframe["humerothoracic_angle"] = concatenated_dataframe[
            [f"humerothoracic_angle_dof{i+1}" for i in dof_idx]
        ].mean(axis=1)

        if drop_humerothoracic_raw_data:
            concatenated_dataframe.drop(
                columns=[f"humerothoracic_angle_dof{i+1}" for i in dof_idx],
                inplace=True,
            )

    # Fill with nans the missing dof
    absent_dof_idx = [i for i in range(0, 3) if i not in dof_idx]
    for j in absent_dof_idx:
        concatenated_dataframe[f"value_dof{j+1}"] = np.nan

    return concatenated_dataframe


def load_csv(csv_filenames, columns):
    """Load the csv file from the filename and return a pandas dataframe."""
    if csv_filenames is not None:
        # print(f"Loading {csv_filenames}")
        csv_file_dof1 = pd.read_csv(csv_filenames, sep=",", header=None)
        csv_file_dof1.columns = columns
    else:
        csv_file_dof1 = pd.DataFrame(columns=columns)

    return csv_file_dof1
