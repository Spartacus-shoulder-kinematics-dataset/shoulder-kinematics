from pathlib import Path

import pandas as pd

from .enums import DatasetCSV, DataFolder
from .row_data import RowData


class Spartacus:
    """
    This is a Dataset Class.
    The class can have methods to load the data, filter it, or perform common operations in a natural language style.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):
        self.dataframe = dataframe

        self.clean_df()
        # self.remove_rows_not_ready_for_analysis() # Todo: remove this function ultimately
        self.rows = []
        self.rows_output = None

        self.corrected_confident = None
        self.corrected_confident_data_values = None
        self.confident_data_values = None

    def clean_df(self):
        # turn nan into None for the following columns
        # dof_1st_euler, dof_2nd_euler, dof_3rd_euler, dof_translation_x, dof_translation_y, dof_translation_z
        self.dataframe = self.dataframe.where(pd.notna(self.dataframe), None)

    def remove_rows_not_ready_for_analysis(self):
        # Todo: remove this function ultimately
        # remove lines I know they are not ready for analysis
        dataset_authors = [
            "Gutierrez Delgado et al.",  # not usable finally ?
            "Kim et al.",  # no data yet.
        ]
        for a in dataset_authors:
            self.dataframe.drop(
                self.dataframe[self.dataframe["dataset_authors"].str.contains(a)].index,
                inplace=True,
            )
        article_author_year = [
            "Sahara et al.",  # no rotation data
            "Sugi et al.",
        ]
        for a in article_author_year:
            self.dataframe.drop(
                self.dataframe[self.dataframe["dataset_authors"].str.contains(a)].index,
                inplace=True,
            )

    def set_correction_callbacks_from_segment_joint_validity(self, print_warnings: bool = False) -> pd.DataFrame:
        """
        This function will add a callback function to the dataframe.
        Before setting the callback function, it will check the validity of the joint and the segments
        declared in the dataframe.

        !!! It skips the rows that are not valid.

        Parameters
        ---------
        print_warnings: bool
            This displays warning when necessary.
        """
        # columns
        columns = self.dataframe.columns

        # create an empty dataframe
        self.confident_dataframe = pd.DataFrame(columns=columns)

        for i, row in self.dataframe.iterrows():

            row_data = RowData(row)
            if print_warnings:
                print("")
                print("")
                print("")
                print("row_data.joint", row.dataset_authors)

            if not row_data.check_all_segments_validity(print_warnings=print_warnings):
                continue

            row_data.set_segments()

            if not row_data.check_joint_validity(print_warnings=print_warnings):
                continue

            if not row_data.check_thoracohumeral_angle(print_warnings=print_warnings):
                continue

            # rotation_validity, translation_validity = row_data.check_segments_correction_validity(
            #     print_warnings=print_warnings
            # )

            if not row_data.has_rotation_data and not row_data.has_translation_data:
                print("WARNING : No usable data for this row, in both rotation and translation...")
                continue

            # if not row_data.has_rotation_data and row_data.has_translation_data:
            #     # Todo: handle translation, NISHINAKA for example
            #     print("WARNING : Only translation handled for this row... not yet working")
            #     continue

            row_data.compute_deviations()
            row_data.set_rotation_correction_callback()

            # add the row to the dataframe
            self.confident_dataframe = pd.concat([self.confident_dataframe, row.to_frame().T], ignore_index=True)

        return self.confident_dataframe

    def import_confident_data(self) -> pd.DataFrame:
        """
        This function will import the data from the dataframe, using the callback functions.
        Only the data corresponding to the rows that are considered good and have a callback function will be imported.
        """
        if self.confident_dataframe is None:
            raise ValueError(
                "The dataframe has not been checked yet. " "Use set_correction_callbacks_from_segment_joint_validity"
            )

        output_dataframe = pd.DataFrame(
            columns=[
                "article",
                "joint",
                "degree_of_freedom",
                "biomechanical_dof",
                "humeral_motion",
                "humerothoracic_angle",
                "value",
                "unit",
                "confidence",
                "shoulder_id",
            ]
        )
        corrected_output_dataframe = output_dataframe.copy()

        for i, row in self.confident_dataframe.iterrows():
            row_data = RowData(row)

            row_data.check_all_segments_validity(print_warnings=False)
            row_data.set_segments()
            row_data.check_joint_validity(print_warnings=False)
            row_data.check_segments_correction_validity(print_warnings=False)
            row_data.check_thoracohumeral_angle(print_warnings=False)
            row_data.set_rotation_correction_callback()

            row_data.import_data()
            row_data.compute_deviations()

            df_series = row_data.to_dataframe(correction=False)
            df_corrected_series = row_data.to_series_dataframe(correction=True)
            # add the row to the dataframe
            output_dataframe = pd.concat([output_dataframe, df_series], ignore_index=True)
            corrected_output_dataframe = pd.concat([corrected_output_dataframe, df_corrected_series], ignore_index=True)

        self.confident_data_values = output_dataframe
        self.corrected_confident_data_values = corrected_output_dataframe

        return self.corrected_confident_data_values

    def export(self):
        path_next_to_clean = Path(DatasetCSV.CLEAN.value).parent

        confident_path = Path.joinpath(path_next_to_clean, "corrected_confident_data.csv")
        self.corrected_confident_data_values.to_csv(confident_path, index=False)

        confident_path = Path.joinpath(path_next_to_clean, "confident_data.csv")
        self.confident_data_values.to_csv(confident_path, index=False)


def load() -> Spartacus:
    """Load the confident dataset"""
    # open the file only_dataset_raw.csv
    df = pd.read_csv(DatasetCSV.CLEAN.value)
    # temporary for debugging
    # df = df[df["dataset_authors"] == "Fung et al."]
    # keep Fung and Bourne
    # df = df[df["dataset_authors"].isin(["Fung et al.", "Bourne"])]
    # df = df[df["dataset_authors"] == "Bourne"]
    # df = df[df["dataset_authors"] == "Chu et al."]
    # df = df[df["dataset_authors"] == "Dal Maso et al."] # expected to be the same
    # df = df[df["dataset_authors"] == "Fung et al."]  # One flipped angle in ST in the middle, looks ok
    # df = df[df["dataset_authors"] == "Kolz et al."]  # expected to shift because of correction for now
    # df = df[df["dataset_authors"] == "Kijima et al."]  # expected some Nan because only one dof for GH
    # df = df[df["dataset_authors"] == "Kozono et al."]
    # df = df[df["dataset_authors"] == "Lawrence et al."]
    # df = df[df["dataset_authors"] == "Matsumura et al."]
    # df = df[df["dataset_authors"] == "Oki et al."]
    # df = df[df["dataset_authors"] == "Teece et al."]
    # df = df[df["dataset_authors"] == "Yoshida et al."]
    # df = df[df["dataset_authors"] != "Nishinaka et al."]
    # df = df[df["dataset_authors"] == "Nishinaka et al."]

    print(df.shape)
    sp = Spartacus(dataframe=df)
    sp.remove_rows_not_ready_for_analysis()
    sp.set_correction_callbacks_from_segment_joint_validity(print_warnings=True)
    sp.import_confident_data()
    # df = load_confident_data(df, print_warnings=True)
    print(df.shape)
    return sp


def load_subdataset(name: DataFolder | str) -> Spartacus:
    """Load the confident dataset"""
    # open the file only_dataset_raw.csv
    df = pd.read_csv(DatasetCSV.CLEAN.value)
    datafolder_string = name if isinstance(name, str) else name.to_dataset_author()
    df = df[df["dataset_authors"] == datafolder_string]
    sp = Spartacus(dataframe=df)
    sp.set_correction_callbacks_from_segment_joint_validity(print_warnings=True)
    sp.import_confident_data()
    return sp
