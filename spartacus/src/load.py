from pathlib import Path

import pandas as pd

from .checks import check_all_segments_validity
from ..enums import DatasetCSV, DataFolder
from .enums_biomech import Segment, JointType
from .row_data import RowData
from .utils import convert_df_to_1dof_per_line

from .checks import check_segment_filled_with_nan
from .compliance import JointCompliance, SegmentCompliance
from .utils import get_segment_columns_direction
from .utils_setters import (
    set_joint_from_row,
    set_thoracohumeral_angle_from_row,
    set_parent_segment_from_row,
    set_child_segment_from_row,
)


class Spartacus:
    """
    A class to represent the Spartacus dataset and its operations.

    Attributes
    ----------
    datasets : pd.DataFrame | None
        DataFrame containing the datasets.
    joint_data : pd.DataFrame | None
        DataFrame containing the joint data.
    unify : bool
        Flag to unify the dataset by checking segments and importing confident data.
    process_rotations : bool
        Flag to process rotations.
    process_translations : bool
        Flag to process translations.
    dataframe : pd.DataFrame
        Merged DataFrame of datasets and joint data.
    confident_dataframe : pd.DataFrame | None
        DataFrame containing confident data.
    rows : list
        List to store rows.
    rows_output : None
        Placeholder for rows output.
    corrected_confident : None
        Placeholder for corrected confident data.
    corrected_confident_data_values : None
        Placeholder for corrected confident data values.
    confident_data_values : None
        Placeholder for confident data values.

    Methods
    -------
    clean_df():
        Replace NaNs with None in the datasets and joint data.
    check_dataset_segments(print_warnings: bool = False) -> pd.DataFrame:
        Check if segments are consistently defined in the dataset.
    import_confident_data() -> pd.DataFrame:
        Import data from the DataFrame using callback functions.
    _add_metadata_to_dataframes():
        Add metadata to the dataframes for further analysis.
    export():
        Export the corrected confident data to the same folder as the clean data.
    compliance() -> pd.DataFrame:
        Calculate compliance for each dataset.
    add_compliances():
        Add compliances to the main dataframe.
    """

    def __init__(
        self,
        datasets: pd.DataFrame | None = None,
        joint_data: pd.DataFrame | None = None,
        unify: bool = False,
        process_rotations: bool = True,
        process_translations: bool = True,
    ):
        """
        Constructs all the necessary attributes for the Spartacus object.

        Parameters
        ----------
        datasets : pd.DataFrame | None, optional
            DataFrame containing the datasets (default is None).
        joint_data : pd.DataFrame | None, optional
            DataFrame containing the joint data (default is None).
        unify : bool, optional
            Flag to unify the dataset by checking segments and importing confident data (default is False).
        process_rotations : bool, optional
            Flag to process rotations (default is True).
        process_translations : bool, optional
            Flag to process translations (default is True).
        """
        self.datasets = datasets
        self.joint_data = joint_data

        # merge the datasets and the joint data through the column dataset_id, dataset_id, joint_data is the bigger file
        self.dataframe = pd.merge(
            datasets, joint_data, left_on="dataset_id", right_on="dataset_id", suffixes=("", "useless_string")
        )
        self.confident_dataframe = None

        self.clean_df()
        self.add_compliances()
        self.rows = []
        self.rows_output = None

        self.corrected_confident = None
        self.corrected_confident_data_values = None
        self.confident_data_values = None

        self.process_rotations = process_rotations
        self.process_translations = process_translations

        if unify:
            self.check_dataset_segments(print_warnings=True)
            self.import_confident_data()

    def clean_df(self):
        """Replace nans with None"""
        self.datasets = self.datasets.where(pd.notna(self.datasets), None)
        self.joint_data = self.joint_data.where(pd.notna(self.joint_data), None)
        self.dataframe = self.dataframe.where(pd.notna(self.dataframe), None)

    def check_dataset_segments(self, print_warnings: bool = False) -> pd.DataFrame:
        """
        This will check if segment are consistently defined in the dataset, with or wihtout nans, direct frames, etc...

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
            suffixes=("", "useless_string"),
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
            ],
        )
        corrected_output_dataframe = output_dataframe.copy()

        for i, row in self.confident_dataframe.iterrows():

            row_data = RowData(row)

            process_translation = row_data.has_translation_data if self.process_translations else False
            process_rotation = row_data.has_rotation_data if self.process_rotations else False

            row_data.set_segments()
            row_data.check_joint_validity(print_warnings=False)
            row_data.check_segments_correction_validity(print_warnings=False)
            row_data.check_thoracohumeral_angle(print_warnings=False)
            row_data.set_compliance()

            if not (process_translation and row_data.enough_compliant_for_translation):
                process_translation = False
            else:
                row_data.set_translation_correction_callback()

            if process_rotation:
                row_data.set_rotation_correction_callback()

            row_data.import_data()

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

        self.confident_data_values = convert_df_to_1dof_per_line(output_dataframe)
        self.corrected_confident_data_values = convert_df_to_1dof_per_line(corrected_output_dataframe)

        self._add_metadata_to_dataframes()

        return self.corrected_confident_data_values

    def _add_metadata_to_dataframes(self):
        """For further analysis, add metadata to the dataframes"""
        meta_data = self.datasets[
            [
                "dataset_authors",
                "in_vivo",
                "experimental_mean",
                "type_of_movement",
                "active",
                "posture",
                "thorax_is_global",
            ]
        ]
        self.confident_data_values = pd.merge(
            self.confident_data_values, meta_data, how="left", left_on="article", right_on="dataset_authors"
        )
        self.confident_data_values = self.confident_data_values.drop(columns="dataset_authors")

        self.corrected_confident_data_values = pd.merge(
            self.corrected_confident_data_values, meta_data, how="left", left_on="article", right_on="dataset_authors"
        )
        self.corrected_confident_data_values = self.corrected_confident_data_values.drop(columns="dataset_authors")

    def export(self):
        """Export the corrected confident data to the same folder as the clean data"""
        path_next_to_clean = Path(DatasetCSV.DATASETS.value).parent

        confident_path = Path.joinpath(path_next_to_clean, "corrected_confident_data.csv")
        self.corrected_confident_data_values.to_csv(confident_path, index=False)

        confident_path = Path.joinpath(path_next_to_clean, "confident_data.csv")
        self.confident_data_values.to_csv(confident_path, index=False)

    @classmethod
    def load(
        cls,
        datasets: DataFolder | str | list[DataFolder | str] = None,
        shoulder: list[int] = None,
        mvt: list[str] | str = None,
        joints: list[str] | str = None,
        unify: bool = True,
        process_rotations: bool = True,
        process_translations: bool = True,
    ):
        """
        Load the confident subdataset

        Parameters
        ----------
        datasets: DataFolder
            The name of the DataFolder, if None keeps everything
        shoulder: list[int]
            The id of the shoulders you want to keep, to study specific data, if None keeps everything
        mvt: list[str] | str
            The shoulder motion of interests to keep, to study specific motions, e.g. sagittal plane elevation,
            if None keeps everything
        joints: list[str] | str
            The joint of interests to keep, to study specific joints, e.g. scapulothoracic
            if None keeps everything
        unify: bool
            If True, the dataset will be unified, i.e. the segments will be checked and the confident data will be imported
            and corrections will be applied.
        process_rotations: bool
            Choose if the rotations should be processed or not.
        process_translations: bool
            Choose if the translations should be processed or not.
        """
        # open the file only_dataset_raw.csv
        df = pd.read_csv(DatasetCSV.DATASETS.value)
        df_joint_data = pd.read_csv(DatasetCSV.JOINT.value)

        if datasets is not None:
            datasets = [datasets] if not isinstance(datasets, list) else datasets
            datafolder_string = [name if isinstance(name, str) else name.to_dataset_author() for name in datasets]
            df = df[df["dataset_authors"].isin(datafolder_string)]
            df_joint_data = df_joint_data[df_joint_data["dataset_authors"].isin(datafolder_string)]

        if shoulder is not None:
            shoulder = [shoulder] if isinstance(shoulder, int) else shoulder
            df_joint_data = df_joint_data[df_joint_data["shoulder_id"].isin(shoulder)]

        if mvt is not None:
            mvt = [mvt] if isinstance(mvt, str) else mvt
            df_joint_data = df_joint_data[df_joint_data["humeral_motion"].isin(mvt)]

        if joints is not None:
            joints = [joints] if isinstance(joints, str) else joints
            df_joint_data = df_joint_data[df_joint_data["joint"].isin(joints)]

        return cls(
            datasets=df,
            joint_data=df_joint_data,
            unify=unify,
            process_rotations=process_rotations,
            process_translations=process_translations,
        )

    @property
    def authors(self):
        """Return the authors of the datasets"""
        return self.dataframe["dataset_authors"].unique().tolist()

    def compliance(self) -> pd.DataFrame:
        """Calculate compliance for each dataset"""
        authors = self.authors

        df_compliance = pd.DataFrame(
            columns=[
                "id",
                "dataset_authors",
                "thorax_c1",
                "thorax_c2",
                "thorax_c3",
                "clavicle_c1",
                "clavicle_c2",
                "clavicle_c3",
                "scapula_c1",
                "scapula_c2",
                "scapula_c3",
                "humerus_c1",
                "humerus_c2",
                "humerus_c3",
                "sternoclavicular_c4",
                "sternoclavicular_c5",
                "acromioclavicular_c4",
                "acromioclavicular_c5",
                "scapulothoracic_c4",
                "scapulothoracic_c5",
                "glenohumeral_c4",
                "glenohumeral_c5",
                "thoracohumeral_c6",
            ]
        )

        # collect joint for which I have data
        df_grouped = self.dataframe.groupby("dataset_authors")["joint"].agg(lambda x: list(set(x))).reset_index()
        joints_per_author = df_grouped.set_index("dataset_authors")["joint"].to_dict()

        for i, author in enumerate(authors):
            print(f"Processing {author} ({i + 1}/{len(authors)})")

            subdf = self.dataframe[self.dataframe["dataset_authors"] == author]
            first_row = subdf.iloc[0]

            dico_d = {}
            dico_d["id"] = first_row["dataset_id"]
            dico_d["dataset_authors"] = first_row["dataset_authors"]

            for segment in Segment:

                segment_cols = get_segment_columns_direction(segment)
                if not check_segment_filled_with_nan(first_row, segment_cols, print_warnings=True):
                    bsys_segment = set_parent_segment_from_row(first_row, segment)
                    compliance = SegmentCompliance(bsys=bsys_segment)
                    dico_d[f"{segment.to_string}_c1"] = compliance.is_c1
                    dico_d[f"{segment.to_string}_c2"] = compliance.is_c2
                    dico_d[f"{segment.to_string}_c3"] = compliance.is_c3
                if (
                    check_segment_filled_with_nan(first_row, segment_cols, print_warnings=True)
                    and first_row[segment_cols[3]] is not None
                ):
                    #  for nishinaka for example that only has translational information
                    bsys_segment = set_child_segment_from_row(first_row, segment)
                    compliance = SegmentCompliance(bsys=bsys_segment)
                    dico_d[f"{segment.to_string}_c3"] = compliance.is_c3

            for joint_type_str in joints_per_author[author]:
                joint_type = JointType.from_string(joint_type_str)

                first_row = subdf[subdf["joint"] == joint_type_str].iloc[0]
                joint = set_joint_from_row(first_row, joint_type)

                thoracohumeral_angle = set_thoracohumeral_angle_from_row(first_row)
                joint_deviation = JointCompliance(joint=joint, thoracohumeral_angle=thoracohumeral_angle)

                if joint.euler_sequence is not None:
                    dico_d[f"{joint_type.to_string}_c4"] = joint_deviation.is_c4
                if joint.translation_origin is not None:
                    dico_d[f"{joint_type.to_string}_c5"] = joint_deviation.is_c5
                dico_d[f"thoracohumeral_c6"] = joint_deviation.is_c6

            df_compliance = pd.concat([df_compliance, pd.DataFrame([dico_d])], ignore_index=True)

        return df_compliance

    def add_compliances(self):
        """It adds the compliances to the main dataframe - self.dataframe"""
        df_compliance = self.compliance()
        df_compliance = df_compliance.drop(columns="dataset_authors")
        self.datasets = pd.merge(
            self.datasets,
            df_compliance,
            left_on="dataset_id",
            right_on="id",
        )
        self.datasets = self.datasets.drop(columns="id")
        self.dataframe = pd.merge(
            self.datasets, self.joint_data, left_on="dataset_id", right_on="dataset_id", suffixes=("", "useless_string")
        )
