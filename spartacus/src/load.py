from pathlib import Path

import pandas as pd

from .checks import check_all_segments_validity
from .enums import DatasetCSV, DataFolder
from .row_data import RowData
from .utils import convert_df_to_1dof_per_line


class Spartacus:
    """
    This is a Dataset Class.
    The class can have methods to load the data, filter it, or perform common operations in a natural language style.
    """

    def __init__(
        self,
        datasets: pd.DataFrame | None = None,
        joint_data: pd.DataFrame | None = None,
    ):
        self.datasets = datasets
        self.joint_data = joint_data
        # merge the datasets and the joint data through the column dataset_id, dataset_id, joint_data is the bigger file
        self.dataframe = pd.merge(datasets, joint_data, left_on="dataset_id", right_on="dataset_id")
        self.confident_dataframe = None

        self.clean_df()
        self.rows = []
        self.rows_output = None

        self.corrected_confident = None
        self.corrected_confident_data_values = None
        self.confident_data_values = None

    def clean_df(self):
        # turn nan into None for the following columns
        # dof_1st_euler, dof_2nd_euler, dof_3rd_euler, dof_translation_x, dof_translation_y, dof_translation_z
        # self.dataframe = self.dataframe.where(pd.notna(self.dataframe), None)
        self.datasets = self.datasets.where(pd.notna(self.datasets), None)
        self.joint_data = self.joint_data.where(pd.notna(self.joint_data), None)

    def check_dataset_segments(self, print_warnings: bool = False) -> pd.DataFrame:
        """
        This will chekc if segment are consistently defined in the dataset, with or wihtout nans, direct frames, etc...

        !!! It skips the rows that are not valid.

        Parameters
        ---------
        print_warnings: bool
            This displays warning when necessary.
        """
        # columns
        columns = self.datasets.columns

        # create an empty dataframe
        self.confident_dataframe = pd.DataFrame(columns=columns)

        for i, row in self.datasets.iterrows():

            if print_warnings:
                print("")
                print("")
                print("row_data.joint", row.dataset_authors)

            if not check_all_segments_validity(row, print_warnings=print_warnings):
                continue

            # add the row to the dataframe
            self.confident_dataframe = pd.concat([self.confident_dataframe, row.to_frame().T], ignore_index=True)

        self.confident_dataframe = pd.merge(
            self.confident_dataframe,
            self.joint_data.drop("dataset_authors", axis=1),
            left_on="dataset_id",
            right_on="dataset_id",
        )
        return self.confident_dataframe

    def import_confident_data(self) -> pd.DataFrame:
        """
        This function will import the data from the dataframe, using the callback functions.
        Only the data corresponding to the rows that are considered good and have a callback function will be imported.
        """
        if self.confident_dataframe is None:
            raise ValueError(
                "The dataframe has not been checked yet. " "Use check_dataset_segments() before importing the data."
            )

        output_dataframe = pd.DataFrame(
            columns=[
                "article",  # string
                "joint",  # string
                "humeral_motion",  # string
                "humerothoracic_angle",  # float
                "value_dof1",  # float
                "value_dof2",  # float
                "value_dof3",  # float
                "legend_dof1",  # string
                "legend_dof2",  # string
                "legend_dof3",  # string
                "unit",  # string "angle" or "translation"
                "shoulder_id",  # int
                "in_vivo",  # bool
                "xp_mean",  # string
            ],
        )
        corrected_output_dataframe = output_dataframe.copy()

        for i, row in self.confident_dataframe.iterrows():

            row_data = RowData(row)

            process_translation = row_data.has_translation_data
            process_rotation = row_data.has_rotation_data

            row_data.set_segments()
            row_data.check_joint_validity(print_warnings=False)
            row_data.check_segments_correction_validity(print_warnings=False)
            row_data.check_thoracohumeral_angle(print_warnings=False)

            if not (process_translation and row_data.enough_compliant_for_translation):
                process_translation = False
            else:
                row_data.set_translation_correction_callback()

            if process_rotation:
                row_data.set_rotation_correction_callback()

            row_data.import_data()
            row_data.compute_deviations()

            df_series = row_data.to_dataframe(
                correction=False,
                translation=process_translation,
                rotation=process_rotation,
            )
            df_corrected_series = row_data.to_dataframe(
                correction=True, translation=process_translation, rotation=process_rotation
            )
            # add the row to the dataframe
            output_dataframe = pd.concat([output_dataframe, df_series], ignore_index=True)
            corrected_output_dataframe = pd.concat([corrected_output_dataframe, df_corrected_series], ignore_index=True)

        # self.confident_data_values = output_dataframe
        # self.corrected_confident_data_values = corrected_output_dataframe
        self.confident_data_values = convert_df_to_1dof_per_line(output_dataframe)
        self.corrected_confident_data_values = convert_df_to_1dof_per_line(corrected_output_dataframe)

        return self.corrected_confident_data_values

    def export(self):
        path_next_to_clean = Path(DatasetCSV.CLEAN.value).parent

        confident_path = Path.joinpath(path_next_to_clean, "corrected_confident_data.csv")
        self.corrected_confident_data_values.to_csv(confident_path, index=False)

        confident_path = Path.joinpath(path_next_to_clean, "confident_data.csv")
        self.confident_data_values.to_csv(confident_path, index=False)


def load() -> Spartacus:
    """Load the confident datasets"""
    # open the file only_dataset_raw.csv
    # df = pd.read_csv(DatasetCSV.CLEAN.value)
    df = pd.read_csv(DatasetCSV.DATASETS.value)
    df_joint_data = pd.read_csv(DatasetCSV.JOINT_DATA.value)
    # temporary for debugging
    # df = df[df["dataset_authors"] == "Fung et al."]
    # keep Fung and Bourne
    # df = df[df["dataset_authors"].isin(["Fung et al.", "Bourne"])]
    # df = df[df["dataset_authors"] == "Bourne"]
    # df = df[df["dataset_authors"] == "Chu et al."]
    # df = df[df["dataset_authors"] == "Henninger et al."]
    # df = df[df["dataset_authors"] == "Fung et al."]  # One flipped angle in ST in the middle, looks ok
    # df = df[df["dataset_authors"] == "Kijima et al."]  # expected some Nan because only one dof for GH
    # df = df[df["dataset_authors"] == "Kozono et al."]
    # df = df[df["dataset_authors"] == "Lawrence et al."]
    # df = df[df["dataset_authors"] == "Matsumura et al."]
    # df = df[df["dataset_authors"] == "Oki et al."]
    # df = df[df["dataset_authors"] == "Teece et al."]
    # df = df[df["dataset_authors"] == "Yoshida et al."]
    # df = df[df["dataset_authors"] != "Nishinaka et al."]
    # df = df[df["dataset_authors"] == "Nishinaka et al."]

    df = df[df["dataset_authors"] != "Gutierrez Delgado et al."]

    print(df.shape)
    sp = Spartacus(datasets=df, joint_data=df_joint_data)
    sp.check_dataset_segments(print_warnings=True)
    sp.import_confident_data()
    # df = load_confident_data(df, print_warnings=True)
    print(df.shape)
    return sp


def load_subdataset(name: DataFolder | str) -> Spartacus:
    """Load the confident dataset"""
    # open the file only_dataset_raw.csv
    df = pd.read_csv(DatasetCSV.CLEAN.value)
    df_joint_data = pd.read_csv(DatasetCSV.JOINT_DATA.value)

    datafolder_string = name if isinstance(name, str) else name.to_dataset_author()
    df = df[df["dataset_authors"] == datafolder_string]

    sp = Spartacus(datasets=df, joint_data=df_joint_data)
    sp.check_dataset_segments(print_warnings=True)
    sp.import_confident_data()
    return sp
