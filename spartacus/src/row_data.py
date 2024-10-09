import os

import numpy as np
import pandas as pd

from ..enums import (
    DataFolder,
)
from .checks import (
    check_segment_filled_with_nan,
    check_is_euler_sequence_provided,
    check_is_translation_provided,
    check_parent_child_joint,
    check_correction_methods,
)
from .compliance import SegmentCompliance, JointCompliance, TotalCompliance
from .constants import REPEATED_DATAFRAME_KEYS
from .corrections.angle_conversion_callbacks import (
    isb_framed_rotation_matrix_from_euler_angles,
    set_corrections_on_rotation_matrix,
    rotation_matrix_2_euler_angles,
    to_left_handed_frame,
    quick_fix_x_rot_in_yxy_if_x_positive,
    quick_fix_x_rot_in_yxy_from_matrix,
    from_euler_angles_to_rotation_matrix,
)
from .corrections.kolz_matrices import get_kolz_rotation_matrix
from .corrections.unwrap_utils import unwrap_for_yxy_glenohumeral_joint
from .corrections.euler_basis import from_jcs_to_parent_frame
from .enums_biomech import (
    Segment,
    FrameType,
    Correction,
    EulerSequence,
    AnatomicalLandmark,
    JointType,
)
from .joint import Joint
from .load_data import load_euler_csv
from .utils import (
    get_segment_columns_direction,
    get_correction_column,
    get_is_isb_column,
    calculate_dof_values,
)
from .utils_setters import set_parent_segment_from_row, set_child_segment_from_row, set_thoracohumeral_angle_from_row


class RowData:
    """
    This class is used to store the data of a row of the dataset and make it accessible through attributes and methods.
    """

    def __init__(self, row: pd.Series):
        """
        Parameters
        ----------
        row : pandas.Series
            The row of the dataset to store.
        """
        self.row = row

        self.parent_segment = Segment.from_string(self.row.parent)
        self.parent_columns = get_segment_columns_direction(self.parent_segment)

        self.child_segment = Segment.from_string(self.row.child)
        self.child_columns = get_segment_columns_direction(self.child_segment)

        self.right_side = row.side_as_right
        self.thoracohumeral_angle = None

        self.parent_biomech_sys = None
        self.parent_corrections = None

        self.child_biomech_sys = None
        self.child_corrections = None

        self.joint = None

        self.parent_compliance = None
        self.child_compliance = None
        self.joint_compliance = None
        self.total_compliance = None

        # indirect attributes
        self._has_rotation_data = None  # has euler sequence
        self._has_translation_data = None  # has coordinate system and origin

        self.rotation_deviation = None
        self.translation_deviation = None

        self.parent_segment_usable_for_rotation_data = None
        self.child_segment_usable_for_rotation_data = None

        self.parent_segment_usable_for_translation_data = None
        self.child_segment_usable_for_translation_data = None

        self.parent_definition_risk = None
        self.child_definition_risk = None

        self.usable_rotation_data = None
        self.usable_translation_data = None

        self.rotation_data_risk = None
        self.translation_data_risk = None

        self.euler_angles_correction_callback = None
        self.translation_correction_callback = None
        self.translation_isb_matrix_callback = None

        self.csv_filenames = None
        self.csv_translation_filenames = None
        self.data = None
        self.df_3dof_per_line = None
        self.df_1dof_per_line = None
        self.corrected_df_3dof_per_line = None
        self.corrected_df_1dof_per_line = None

    @property
    def has_rotation_data(self) -> bool:
        columns = ["dof_1st_euler", "dof_2nd_euler", "dof_3rd_euler"]
        return any([self.row[column] is not None for column in columns])

    @property
    def has_translation_data(self) -> bool:
        columns = ["dof_translation_x", "dof_translation_y", "dof_translation_z"]
        return any([self.row[column] is not None for column in columns])

    @property
    def left_side(self):
        return not self.right_side

    def check_thoracohumeral_angle(self, print_warnings: bool = False):
        """
        Check if the thoracohumeral angle is well-defined in the database and set the attributes

        Returns
        -------
        bool
            True if the thoracohumeral angle is valid, False otherwise.
        """
        output = True

        if self.row.thoracohumeral_sequence == "nan":
            output = False
            if print_warnings:
                print(f"Joint {self.row.joint} has no thoracohumeral angle defined, " f"it should not be empty !!!")
            return output

        if self.row.thoracohumeral_sequence == "nan" and self.row.thoracohumeral_angle == "nan":
            output = False
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has no thoracohumeral sequence defined, "
                    f" and no thoracohumeral angle defined, "
                    f"it should not be empty !!!"
                )
            return output

        self.thoracohumeral_angle = set_thoracohumeral_angle_from_row(self.row)

        return output

    def check_joint_validity(self, print_warnings: bool = False) -> bool:
        """
        Check if the joint defined in the dataset is valid.
        We expect the joint to have a valid euler sequence, i.e., no NaN values., three letters and valid letters.
        If not we expect the joint to have a valid translation, i.e., no NaN values.

        We expect the joint to have good parent and child definitions
        We expect the joint to have defined parent and child segments, i.e., no NaN values.

        Returns
        -------
        bool
            True if the joint is valid, False otherwise.
        """
        output = True

        # todo: separate as much as possible the rotations checks and the translations checks

        no_euler_sequence = not check_is_euler_sequence_provided(self.row, print_warnings=print_warnings)
        no_translation = not check_is_translation_provided(self.row, print_warnings=print_warnings)

        self._has_rotation_data = not no_euler_sequence
        self._has_translation_data = not no_translation

        if no_euler_sequence and no_translation:
            output = False
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has no euler sequence defined, "
                    f" and no translation defined, "
                    f"it should not be empty !!!"
                    f" Got euler sequence: {self.row.euler_sequence}, "
                    f"Got translation: {self.row.origin_displacement}, {self.row.displacement_cs}"
                )
            return output

        self.set_joint(no_euler_sequence=no_euler_sequence, no_translation=no_translation)

        if not check_parent_child_joint(self.joint, row=self.row, print_warnings=print_warnings):
            output = False

        # check database if nan in one the segment of the joint
        if check_segment_filled_with_nan(self.row, self.parent_columns, print_warnings=print_warnings):
            output = False
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has a NaN value in the parent segment {self.row.parent}, "
                    f"it should not be empty !!!"
                )

        if check_segment_filled_with_nan(self.row, self.child_columns, print_warnings=print_warnings):
            output = False
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has a NaN value in the child segment {self.row.child}, "
                    f"it should not be empty !!!"
                )

        return output

    def set_joint(self, no_euler_sequence: bool = None, no_translation: bool = None):
        if no_euler_sequence:  # Only translation is provided
            self.joint = Joint(
                joint_type=JointType.from_string(self.row.joint),
                euler_sequence=EulerSequence.from_string(self.row.euler_sequence),  # throw a None
                translation_origin=AnatomicalLandmark.from_string(self.row.origin_displacement),
                translation_frame=FrameType.from_string(self.row.displacement_cs),
                parent_segment=self.parent_biomech_sys,
                child_segment=self.child_biomech_sys,
            )

        elif no_translation:  # Only rotation is provided
            self.joint = Joint(
                joint_type=JointType.from_string(self.row.joint),
                euler_sequence=EulerSequence.from_string(self.row.euler_sequence),
                translation_origin=None,
                translation_frame=None,
                parent_segment=self.parent_biomech_sys,
                child_segment=self.child_biomech_sys,
            )

        else:  # translation and rotation are both provided
            self.joint = Joint(
                joint_type=JointType.from_string(self.row.joint),
                euler_sequence=EulerSequence.from_string(self.row.euler_sequence),
                translation_origin=AnatomicalLandmark.from_string(self.row.origin_displacement),
                translation_frame=FrameType.from_string(self.row.displacement_cs),
                parent_segment=self.parent_biomech_sys,
                child_segment=self.child_biomech_sys,
            )

    def set_segments(self):
        """
        Set the parent and child segments of the joint.
        """
        self.parent_biomech_sys = set_parent_segment_from_row(self.row, self.parent_segment)
        self.child_biomech_sys = set_child_segment_from_row(self.row, self.child_segment)

    def set_compliance(self):
        self.parent_compliance = SegmentCompliance(bsys=self.parent_biomech_sys)
        self.child_compliance = SegmentCompliance(bsys=self.child_biomech_sys)

        thoracohumeral_angle = set_thoracohumeral_angle_from_row(self.row)
        self.joint_compliance = JointCompliance(joint=self.joint, thoracohumeral_angle=thoracohumeral_angle)
        self.total_compliance = TotalCompliance(
            parent_compliance=self.parent_compliance,
            child_compliance=self.child_compliance,
            joint_compliance=self.joint_compliance,
        )

    def extract_corrections(self, segment: Segment) -> str:
        """
        Extract the correction cell of the correction column.
        ex: if the correction column is parent_to_isb, we extract the correction cell parent_to_isb
        """

        correction_column = get_correction_column(segment)
        correction_cell = self.row[correction_column]

        if correction_cell == "nan":
            correction_cell = None
        if not isinstance(correction_cell, str) and correction_cell is not None:
            if np.isnan(correction_cell):
                correction_cell = None

        if correction_cell is not None:
            # separate strings with a comma in several element of list
            correction_cell = correction_cell.replace(" ", "").split(",")
            for i, correction in enumerate(correction_cell):
                correction_cell[i] = Correction.from_string(correction)

        return correction_cell

    def extract_is_thorax_global(self, segment: Segment) -> bool:
        if segment != Segment.THORAX:
            raise ValueError("The segment is not the thorax")
        else:
            return self.row["thorax_is_global"]

    def _check_segment_has_no_correction(self, correction, print_warnings: bool = False) -> bool:
        if correction is not None:
            output = False
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has a correction value in the child segment {self.row.parent}, "
                    f"it should be empty !!!, because the segment is isb. "
                    f"Parent correction: {correction}"
                )
        else:
            output = True
        return output

    def check_segments_correction_validity(self, print_warnings: bool = False) -> tuple[bool, bool]:
        """
        We expect the correction columns to be filled with valid values. Legacy code.

        Return
        ------
        output:tuple[bool, bool]
            rotation_data_validity, translation_data_validity
        """
        parent_output = True
        child_output = True

        check_correction_methods(self, self.parent_biomech_sys)
        check_correction_methods(self, self.child_biomech_sys)

        self.child_corrections = self.extract_corrections(self.child_segment)
        self.parent_corrections = self.extract_corrections(self.parent_segment)

        parent_is_thorax_global = False

        # Thorax is global check
        if self.parent_segment == Segment.THORAX:
            if self.extract_is_thorax_global(self.parent_segment):
                parent_is_thorax_global = True

                self.parent_segment_usable_for_rotation_data = parent_output
                self.parent_segment_usable_for_translation_data = False
                self.parent_definition_risk = True
            else:
                parent_is_thorax_global = False

        # if both segments are isb oriented, but origin is on an isb axis, we expect no correction be filled
        # so that we can consider rotation data as isb
        if (
            self.parent_biomech_sys.is_isb_oriented
            and self.parent_biomech_sys.is_origin_on_an_isb_axis()
            and not parent_is_thorax_global
        ):
            parent_output = self._check_segment_has_no_correction(
                self.parent_corrections, print_warnings=print_warnings
            )
            self.parent_segment_usable_for_rotation_data = parent_output
            self.parent_segment_usable_for_translation_data = False

        if self.child_biomech_sys.is_isb_oriented and self.child_biomech_sys.is_origin_on_an_isb_axis():
            child_output = self._check_segment_has_no_correction(self.child_corrections, print_warnings=print_warnings)
            self.child_segment_usable_for_rotation_data = child_output
            self.child_segment_usable_for_translation_data = False

        if (
            self.parent_biomech_sys.is_isb_oriented
            and not self.parent_biomech_sys.is_origin_on_an_isb_axis()
            and not parent_is_thorax_global
        ):
            self.parent_segment_usable_for_rotation_data = True
            self.parent_segment_usable_for_translation_data = False

        if self.child_biomech_sys.is_isb_oriented and not self.child_biomech_sys.is_origin_on_an_isb_axis():
            child_output = True
            if self.child_segment == Segment.SCAPULA:
                child_output = True
            else:
                self.child_definition_risk = True
            self.child_segment_usable_for_rotation_data = child_output
            self.child_segment_usable_for_translation_data = False

        # if segments are not isb, we expect the correction to_isb to be filled
        if (
            not self.parent_biomech_sys.is_isb_oriented
            and self.parent_biomech_sys.is_origin_on_an_isb_axis()
            and not parent_is_thorax_global
        ):
            parent_output = True
            self.parent_segment_usable_for_rotation_data = parent_output
            self.parent_segment_usable_for_translation_data = False

        if not self.child_biomech_sys.is_isb_oriented and self.child_biomech_sys.is_origin_on_an_isb_axis():
            child_output = True
            self.child_segment_usable_for_rotation_data = child_output
            self.child_segment_usable_for_translation_data = False

        if (
            not self.parent_biomech_sys.is_isb_oriented
            and not self.parent_biomech_sys.is_origin_on_an_isb_axis()
            and not parent_is_thorax_global
        ):
            parent_output = True
            if self.parent_segment == Segment.SCAPULA:
                self.parent_segment_usable_for_rotation_data = parent_output
                self.parent_segment_usable_for_translation_data = False
                self.parent_definition_risk = True  # should be a less high risk. because known from the literature
            else:
                parent_output = True

                self.parent_segment_usable_for_rotation_data = parent_output
                self.parent_segment_usable_for_translation_data = False
                self.parent_definition_risk = True

        if not self.child_biomech_sys.is_isb_oriented and not self.child_biomech_sys.is_origin_on_an_isb_axis():
            child_output = True
            if self.child_segment == Segment.SCAPULA:
                self.child_segment_usable_for_rotation_data = child_output
                self.child_segment_usable_for_translation_data = False
                self.child_definition_risk = True  # should be a less high risk. because known from the literature
            else:
                child_output = True

                self.parent_segment_usable_for_rotation_data = child_output
                self.parent_segment_usable_for_translation_data = False
                self.parent_definition_risk = True

        # finally check the combination of parent and child to determine if usable for rotation and translation
        self.usable_rotation_data = (
            self.child_segment_usable_for_rotation_data and self.parent_segment_usable_for_rotation_data
        )
        self.usable_translation_data = (
            self.child_segment_usable_for_translation_data and self.parent_segment_usable_for_translation_data
        )

        return self.usable_rotation_data, self.usable_translation_data

    def set_rotation_correction_callback(self):
        """
        The idea is to prepare a function ready to receive 3 Euler Angles (rot1, rot2, rot3) from any Euler Sequence,
        and from this sequence:
        - Rebuild the corresponding rotation matrix R_proximal_distal
        - Convert into a rotation matrix into x antero-posterior, y infero-superior, z medio-lateral (right)
        - Switch to a left-handed coordinate system
        if the data are on the left side to have the sign as for the right side on Euler angles
        - Apply a correction if any to make it ISB
        - Convert back into the Euler Sequence

        More mathematically:
        - 1st : R_proximal_distal = R(rot1, rot2, rot3, euler_sequence)
        - 2nd : R_proximal_distal = R_parent_correction @ R_distal_proximal @ R_child_correction (now z is medio-lateral)
        - 3rd if left side : R_proximal_distal = np.diag([1, 1, -1]) @ R_proximal_distal @ np.diag([1, 1, -1]) (now z is medio-lateral, for left side too)
        - 4th : R_proximal_distal = R_parent_correction @ R_proximal_distal @ R_child_correction
        - 5th : rot1, rot2, rot3 = euler_angles(R_proximal_distal, euler_sequence)

        """
        self.isb_rotation_matrix_callback = lambda rot1, rot2, rot3: isb_framed_rotation_matrix_from_euler_angles(
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            previous_sequence_str=self.joint.euler_sequence.value,
            bsys_parent=self.parent_biomech_sys,
            bsys_child=self.child_biomech_sys,
        )

        if self.left_side:
            self.mediolateral_matrix = lambda rot1, rot2, rot3: to_left_handed_frame(
                self.isb_rotation_matrix_callback(rot1=rot1, rot2=rot2, rot3=rot3)
            )
        else:
            self.mediolateral_matrix = self.isb_rotation_matrix_callback

        parent_matrix_correction = (
            np.eye(3)
            if self.parent_corrections is None
            else get_kolz_rotation_matrix(correction=self.parent_corrections[0])
        )
        child_matrix_correction = (
            np.eye(3)
            if self.child_corrections is None
            else get_kolz_rotation_matrix(correction=self.child_corrections[0])
        )

        self.correct_isb_rotation_matrix_callback = lambda rot1, rot2, rot3: set_corrections_on_rotation_matrix(
            matrix=self.mediolateral_matrix(rot1, rot2, rot3),
            child_matrix_correction=child_matrix_correction,
            parent_matrix_correction=parent_matrix_correction,
        )

        self.euler_angles_correction_callback = lambda rot1, rot2, rot3: rotation_matrix_2_euler_angles(
            rotation_matrix=self.correct_isb_rotation_matrix_callback(rot1, rot2, rot3),
            euler_sequence=self.joint.isb_euler_sequence,
        )

        # enforce negative elevation
        if self.joint.joint_type == JointType.GLENO_HUMERAL and self.row.humeral_motion in (
            "scapular plane elevation",
            "sagittal plane elevation",
            "frontal plane elevation",
            "internal-external rotation 0 degree-abducted",
            "internal-external rotation 90 degree-abducted",
            "horizontal flexion",
        ):
            self.euler_angles_correction_callback = lambda rot1, rot2, rot3: quick_fix_x_rot_in_yxy_if_x_positive(
                rotation_matrix_2_euler_angles(
                    rotation_matrix=self.correct_isb_rotation_matrix_callback(rot1, rot2, rot3),
                    euler_sequence=self.joint.isb_euler_sequence,
                ),
            )

    def set_translation_correction_callback(self):
        """
        Work in Progress but here is the idea.

        We want to express the translation in the proximal segment coordinate system in ISB frame
        on the right side.

        It only fixes the ISB orientation, restoring x as antero-posterior, y as infero-superior, z as medio-lateral.
        and the side of the coordinate system left to right if needed.

        Missing features:
        - transport local to distal SCS ?

        """
        if self.joint.translation_frame == FrameType.JCS:
            self.proximal_translation = lambda trans_x, trans_y, trans_z, rot1, rot2, rot3: from_jcs_to_parent_frame(
                np.array([trans_x, trans_y, trans_z]), np.array([rot1, rot2, rot3]), self.joint.euler_sequence
            )
        else:
            self.proximal_translation = lambda trans_x, trans_y, trans_z, rot1, rot2, rot3: np.eye(3) @ np.array(
                [trans_x, trans_y, trans_z]
            )

        self.translation_isb_matrix_callback = (
            lambda trans_x, trans_y, trans_z, rot1, rot2, rot3: self.parent_biomech_sys.get_rotation_matrix()
            @ self.proximal_translation(trans_x, trans_y, trans_z, rot1, rot2, rot3)
        )

        if self.left_side:
            self.translation_mediolateral_matrix = (
                lambda trans_x, trans_y, trans_z, rot1, rot2, rot3: self.translation_isb_matrix_callback(
                    trans_x, trans_y, trans_z, rot1, rot2, rot3
                )
                * np.array([1, 1, -1])
            )
        else:
            self.translation_mediolateral_matrix = self.translation_isb_matrix_callback

    @property
    def enough_compliant_for_translation(self) -> bool:
        """Check if the segment is compliant enough for merging translation data"""
        pc1 = self.parent_compliance.is_c1
        pc2 = self.parent_compliance.is_c2
        pc3 = self.parent_compliance.is_c3
        cc3 = self.child_compliance.is_c3
        c5 = self.joint_compliance.is_c5

        isb_origins = pc3 and cc3
        any_origin_is_wrong = not pc3 or not cc3

        if any_origin_is_wrong:
            return False

        all_good = pc1 and pc2 and isb_origins
        orientation_correction = pc2 and isb_origins  # flipping axis or changing the sign of the axis
        accepted_orientation_offset = pc1 and isb_origins

        # NOTE: Relaxing constraints could accept without pc1 and pc2
        segment_conditions = all_good or orientation_correction or accepted_orientation_offset

        if segment_conditions and c5:
            return True

        if segment_conditions and not c5:
            if self.joint.translation_frame == FrameType.JCS:
                return True
            else:
                return False

    def import_data(self):
        """this function import the data of the following row"""
        print(
            f" Importing data ...\n"
            f" for article {self.row.dataset_authors},"
            f" joint {self.row.joint},"
            f" motion {self.row.humeral_motion},"
            f" subject {self.row.shoulder_id}"
        )
        # load the csv file
        self.csv_filenames = self.get_euler_csv_filenames()
        self.rotation_data = load_euler_csv(self.csv_filenames)

        corrections = self.get_manual_corrections()
        self.rotation_data["value_dof1"] = self.rotation_data["value_dof1"].apply(lambda x: x * corrections[0])
        self.rotation_data["value_dof2"] = self.rotation_data["value_dof2"].apply(lambda x: x * corrections[1])
        self.rotation_data["value_dof3"] = self.rotation_data["value_dof3"].apply(lambda x: x * corrections[2])

        self.csv_translation_filenames = self.get_translation_csv_filenames()
        self.translation_data = load_euler_csv(self.csv_translation_filenames)

        self.rotation_data["article"] = self.row.dataset_authors
        self.rotation_data["joint"] = JointType.from_string(self.row.joint)
        self.rotation_data["humeral_motion"] = self.row.humeral_motion
        self.rotation_data["unit"] = "rad"

        self.translation_data["article"] = self.row.dataset_authors
        self.translation_data["joint"] = JointType.from_string(self.row.joint)
        self.translation_data["humeral_motion"] = self.row.humeral_motion
        self.translation_data["unit"] = "mm"

    def to_dataframe(self, correction: bool = True, rotation: bool = True, translation: bool = True) -> pd.DataFrame:
        """
        This converts the row to a panda dataframe with the angles in degrees with the following columns:
            - article
            - joint
            - angle_translation
            - degree_of_freedom
            - movement
            - humerothoracic_angle (one line per angle)
            - value

        Parameters
        ----------
        correction : bool, optional
            If True, apply the correction, by default True
        rotation : bool, optional
            If True, import rotation data, by default True
        translation : bool, optional
            If True, import translation data, by default True

        Returns
        -------
        pandas.DataFrame
            The dataframe with the data
        """
        to_concat_3dof = []
        prefix = f"{"corrected_" if correction else ""}df"

        if rotation:
            self.to_series_dataframe(correction=correction, rotation=True)
            to_concat_3dof.append(getattr(self, f"{prefix}_rotation_3dof_per_line"))

        if translation:
            self.to_series_dataframe(correction=correction, rotation=False)
            to_concat_3dof.append(getattr(self, f"{prefix}_translation_3dof_per_line"))

        setattr(
            self,
            f"{prefix}_3dof_per_line",
            pd.concat(
                to_concat_3dof if len(to_concat_3dof) >= 1 else [get_empty_series_dataframe()],
            ),
        )
        return self.corrected_df_3dof_per_line if correction else self.df_3dof_per_line

    def to_series_dataframe(self, correction: bool = True, rotation: bool = None) -> pd.DataFrame:
        """
        This converts the row to a panda dataframe with the angles in degrees with the following columns:
         - article
         - joint
         - angle_translation
         - degree_of_freedom
         - movement
         - humerothoracic_angle (one line per angle)
         - value

        Returns
        -------
        pandas.DataFrame
            The dataframe with the angles in degrees
        """
        series_dataframe = get_empty_series_dataframe()

        data = self.rotation_data if rotation else self.translation_data
        prefix = f"{"corrected_" if correction else ""}df_{"rotation" if rotation else "translation"}"

        if data.empty:
            setattr(self, f"{prefix}_3dof_per_line", series_dataframe)
            return series_dataframe

        no_correction_legend = ("x", "y", "z") if not rotation else tuple(self.joint.euler_sequence.value)
        correction_legend = ("x", "y", "z") if not rotation else self.joint.isb_rotation_biomechanical_dof
        three_dof_legend = correction_legend if correction else no_correction_legend
        if correction:
            value_dof = calculate_dof_values(
                data,
                correction_callable=(
                    self.apply_correction_in_radians if rotation else self.apply_correction_to_translation
                ),
                rotation=rotation,
                rotation_data=None if rotation else self.df_3dof_per_line,
            )
            series_dataframe["value_dof1"] = value_dof[:, 0] if correction else data["value_dof1"]
            series_dataframe["value_dof2"] = value_dof[:, 1] if correction else data["value_dof2"]
            series_dataframe["value_dof3"] = value_dof[:, 2] if correction else data["value_dof3"]
        else:
            series_dataframe["value_dof1"] = data["value_dof1"]
            series_dataframe["value_dof2"] = data["value_dof2"]
            series_dataframe["value_dof3"] = data["value_dof3"]

        series_dataframe["legend_dof1"] = three_dof_legend[0]
        series_dataframe["legend_dof2"] = three_dof_legend[1]
        series_dataframe["legend_dof3"] = three_dof_legend[2]
        series_dataframe["humerothoracic_angle"] = data["humerothoracic_angle"]
        series_dataframe["unit"] = data["unit"]

        series_dataframe["article"] = self.row.dataset_authors
        series_dataframe["joint"] = self.row.joint
        series_dataframe["humeral_motion"] = self.row.humeral_motion

        series_dataframe["shoulder_id"] = self.row.shoulder_id

        series_dataframe["parent_compliance_1"] = self.total_compliance.parent.is_c1
        series_dataframe["parent_compliance_2"] = self.total_compliance.parent.is_c2
        series_dataframe["parent_compliance_3"] = self.total_compliance.parent.is_c3
        series_dataframe["child_compliance_1"] = self.total_compliance.child.is_c1
        series_dataframe["child_compliance_2"] = self.total_compliance.child.is_c2
        series_dataframe["child_compliance_3"] = self.total_compliance.child.is_c3
        series_dataframe["joint_compliance_4"] = self.total_compliance.joint.is_c4
        series_dataframe["joint_compliance_5"] = self.total_compliance.joint.is_c5
        series_dataframe["joint_compliance_6"] = self.total_compliance.joint.is_c6
        series_dataframe["total_compliance"] = (
            self.total_compliance.rotation if rotation else self.total_compliance.translation
        )
        series_dataframe["fully_isb"] = (
            self.total_compliance.is_rotation_isb if rotation else self.total_compliance.is_translation_isb
        )
        # remove row where value_dof1, value_dof2 and value_dof3 are NaN
        series_dataframe = series_dataframe.dropna(subset=["value_dof1", "value_dof2", "value_dof3"], how="all")
        setattr(self, f"{prefix}_3dof_per_line", series_dataframe)

        return getattr(self, f"{prefix}_3dof_per_line")

    def get_euler_csv_filenames(self) -> tuple[str, str, str]:
        """load the csv filenames from the row data"""
        folder_path = DataFolder.from_string(self.row["folder"]).value

        csv_paths = ()

        for field in [
            "dof_1st_euler",
            "dof_2nd_euler",
            "dof_3rd_euler",
        ]:
            csv_paths += (os.path.join(folder_path, self.row[field]),) if self.row[field] is not None else (None,)

        return csv_paths

    def get_manual_corrections(self) -> tuple[int, int, int]:
        """load the raw corrections.csv applied on raw data, because we suspect the initial computation to be done wrong"""
        folder_path = DataFolder.from_string(self.row["folder"]).value
        # check if correction.csv is in the folder
        manual_correction = [1, 1, 1]
        correction_csv = "corrections.csv"
        if correction_csv in os.listdir(folder_path):
            correction_csv_path = os.path.join(folder_path, correction_csv)
            correction_df = pd.read_csv(correction_csv_path, sep=",", header=None)
            correction_df.columns = [
                "csv",
                "coefficient",
            ]
            for i, field in enumerate(["dof_1st_euler", "dof_2nd_euler", "dof_3rd_euler"]):
                subdf = correction_df[correction_df["csv"] == self.row[field]]
                if subdf.shape[0] == 1:
                    manual_correction[i] = subdf["coefficient"].values[0]

        return tuple(manual_correction)

    def get_translation_csv_filenames(self) -> tuple[str, str, str]:
        """load the csv filenames from the row data"""
        folder_path = DataFolder.from_string(self.row["folder"]).value

        csv_paths = ()

        for field in [
            "dof_translation_x",
            "dof_translation_y",
            "dof_translation_z",
        ]:
            csv_paths += (os.path.join(folder_path, self.row[field]),) if self.row[field] is not None else (None,)

        return csv_paths

    def apply_correction_in_radians(self, dof1, dof2, dof3) -> tuple[float, float, float]:
        """Apply the correction to the angles in radians"""

        rad_value_dof1 = np.deg2rad(dof1)
        rad_value_dof2 = np.deg2rad(dof2)
        rad_value_dof3 = np.deg2rad(dof3)

        corrected_dof_1, corrected_dof_2, corrected_dof_3 = self.euler_angles_correction_callback(
            rad_value_dof1, rad_value_dof2, rad_value_dof3
        )

        deg_corrected_dof_1 = np.rad2deg(corrected_dof_1)
        deg_corrected_dof_2 = np.rad2deg(corrected_dof_2)
        deg_corrected_dof_3 = np.rad2deg(corrected_dof_3)

        return deg_corrected_dof_1, deg_corrected_dof_2, deg_corrected_dof_3

    def apply_correction_to_translation(self, dof1, dof2, dof3, rot1, rot2, rot3) -> tuple[float, float, float]:
        """Apply the correction to the translation in mm, as we use a matrix product, we need nan to be zeros"""

        dof1 = dof1 if not np.isnan(dof1) else 0
        dof2 = dof2 if not np.isnan(dof2) else 0
        dof3 = dof3 if not np.isnan(dof3) else 0

        rad_value_dof1 = np.deg2rad(rot1) if rot1 is not None else 0
        rad_value_dof2 = np.deg2rad(rot2) if rot2 is not None else 0
        rad_value_dof3 = np.deg2rad(rot3) if rot3 is not None else 0

        corrected_dof_1, corrected_dof_2, corrected_dof_3 = self.translation_mediolateral_matrix(
            dof1, dof2, dof3, rad_value_dof1, rad_value_dof2, rad_value_dof3
        )

        deg_corrected_dof_1 = corrected_dof_1 if corrected_dof_1 != 0 else np.nan
        deg_corrected_dof_2 = corrected_dof_2 if corrected_dof_2 != 0 else np.nan
        deg_corrected_dof_3 = corrected_dof_3 if corrected_dof_3 != 0 else np.nan

        return deg_corrected_dof_1, deg_corrected_dof_2, deg_corrected_dof_3


def get_empty_series_dataframe():
    return pd.DataFrame(
        columns=REPEATED_DATAFRAME_KEYS
        + [
            "humerothoracic_angle",  # float
            "value_dof1",  # float
            "value_dof2",  # float
            "value_dof3",  # float
            "legend_dof1",  # string
            "legend_dof2",  # string
            "legend_dof3",  # string
        ],
    )
