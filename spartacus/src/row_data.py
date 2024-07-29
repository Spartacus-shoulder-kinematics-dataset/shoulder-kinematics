import os

import numpy as np
import pandas as pd

from src.enums import (
    DataFolder,
)
from .checks import (
    check_segment_filled_with_nan,
    check_is_euler_sequence_provided,
    check_is_translation_provided,
    check_parent_child_joint,
)
from .compliance import SegmentCompliance, JointCompliance
from .corrections.angle_conversion_callbacks import (
    isb_framed_rotation_matrix_from_euler_angles,
    set_corrections_on_rotation_matrix,
    rotation_matrix_2_euler_angles,
    to_left_handed_frame,
)
from .corrections.kolz_matrices import get_kolz_rotation_matrix
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
    get_is_correctable_column,
    get_is_isb_column,
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

        self.joint = None
        self.right_side = row.side_as_right
        self.thoracohumeral_angle = None

        self.parent_biomech_sys = None
        self.parent_corrections = None

        self.child_biomech_sys = None
        self.child_corrections = None

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

    def check_all_segments_validity(self, print_warnings: bool = False) -> bool:
        """
        Check all the segments of the row are valid.
        First, we check if the segment is provided, i.e., no NaN values.
        Second, we check if the segment defined as is_isb = True or False in the dataset
        and if the orientations of axis defined in the dataset fits with isb definition.

        (we don't mind if it's not a isb segment, we just don't want to have a segment
        that matches the is_isb given)

        Third, we check the frame are direct, det(R) = 1. We want to have a direct frame.

        Returns
        -------
        bool
            True if all the segments are valid, False otherwise.
        """
        output = True
        for segment_enum in Segment:
            # segment_cols = get_segment_columns(segment_enum)
            segment_cols_direction = get_segment_columns_direction(segment_enum)
            # first check
            if check_segment_filled_with_nan(self.row, segment_cols_direction, print_warnings=print_warnings):
                continue

            bsys = set_parent_segment_from_row(self.row, segment_enum)

            # third check if the segment is direct or not
            if not bsys.is_direct():
                if print_warnings:
                    print(
                        f"{self.row.dataset_authors}, "
                        f"Segment {segment_enum.value} is not direct, "
                        f"it should be !!!"
                    )
                output = False

        return output

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

    def extract_is_correctable(self, segment: Segment) -> bool:
        """
        Extract the database entry to state if the segment is correctable or not.
        """

        if self.row[get_is_correctable_column(segment)] is not None and np.isnan(
            self.row[get_is_correctable_column(segment)]
        ):
            return None
        if self.row[get_is_correctable_column(segment)] == "nan":
            return None
        if self.row[get_is_correctable_column(segment)] == "true":
            return True
        if self.row[get_is_correctable_column(segment)] == "false":
            return False
        if self.row[get_is_correctable_column(segment)]:
            return True
        if not self.row[get_is_correctable_column(segment)]:
            return False

        raise ValueError("The is_correctable column is not a boolean value")

    def extract_is_isb(self, segment: Segment) -> bool:
        """Extract the database entry to state if the segment is isb or not."""
        if self.row[get_is_isb_column(segment)] is not None and np.isnan(self.row[get_is_isb_column(segment)]):
            return None
        if self.row[get_is_isb_column(segment)] == "nan":
            return None
        if self.row[get_is_isb_column(segment)] == "true":
            return True
        if self.row[get_is_isb_column(segment)] == "false":
            return False
        if self.row[get_is_isb_column(segment)]:
            return True
        if not self.row[get_is_isb_column(segment)]:
            return False

        raise ValueError("The is_isb column is not a boolean value")

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

    def _check_segment_has_kolz_correction(self, correction, print_warnings: bool = False) -> bool:
        correction = [] if correction is None else correction
        condition_scapula = (
            Correction.SCAPULA_KOLZ_AC_TO_PA_ROTATION in correction
            or Correction.SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION in correction
        )
        if not condition_scapula:
            output = False
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has no correction value in the segment Scapula, "
                    f"it should be filled with a {Correction.SCAPULA_KOLZ_AC_TO_PA_ROTATION} or a "
                    f"{Correction.SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION} correction, because the segment "
                    f"origin is not on an isb axis. "
                    f"Current value: {correction}"
                )
        else:
            output = True
        return output

    def _check_segment_has_to_isb_correction(self, correction, print_warnings: bool = False) -> bool:
        correction = [] if correction is None else correction
        if not (Correction.TO_ISB_ROTATION in correction):
            output = False
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has no correction value in the parent segment {self.row.parent}, "
                    f"it should be filled with a {Correction.TO_ISB_ROTATION}, because the segment is not isb. "
                    f"Current value: {correction}"
                )
        else:
            output = True
        return output

    def _check_segment_has_to_isb_like_correction(self, correction, print_warnings: bool = False) -> bool:
        correction = [] if correction is None else correction
        if not (Correction.TO_ISB_LIKE_ROTATION in correction):
            output = False
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has no correction value in the parent segment {self.row.parent}, "
                    f"it should be filled with a "
                    f"{Correction.TO_ISB_LIKE_ROTATION} correction, because the segment is not isb. "
                    f"Current value: {correction}"
                )
        else:
            output = True
        return output

    def _check_segment_has_to_isb_or_like_correction(self, correction, print_warnings: bool = False) -> bool:
        correction = [] if correction is None else correction
        output = self._check_segment_has_to_isb_like_correction(correction, print_warnings=False)
        if not output:
            output = self._check_segment_has_to_isb_correction(correction, print_warnings=False)
        if not output:
            if print_warnings:
                print(
                    f"Joint {self.row.joint} has no correction value in the parent segment {self.row.parent}, "
                    f"it should be filled with a "
                    f"{Correction.TO_ISB_LIKE_ROTATION} or {Correction.TO_ISB_ROTATION} "
                    f"correction, because the segment is not isb. "
                    f"Current value: {correction}"
                )
        return output

    def check_segments_correction_validity(self, print_warnings: bool = False) -> tuple[bool, bool]:
        """
        We expect the correction columns to be filled with valid values.
        ex: if both segment are not isb, we expect the correction to_isb to be filled
        ex: if both segment are isb, we expect no correction to be filled
        ex: if both segment are isb, and euler sequence is isb, we expect no correction to be filled
        ex: if both segment are isb, and euler sequence is not isb, we expect the correction to_isb to be filled
        etc...

        Return
        ------
        output:tuple[bool, bool]
            rotation_data_validity, translation_data_validity
        """
        parent_output = True
        child_output = True

        parent_correction = self.extract_corrections(self.parent_segment)
        self.parent_corrections = self.extract_corrections(self.parent_segment)
        parent_is_correctable = self.extract_is_correctable(self.parent_segment)
        parent_is_thorax_global = False

        child_correction = self.extract_corrections(self.child_segment)
        self.child_corrections = self.extract_corrections(self.child_segment)
        # child_is_correctable = self.extract_is_correctable(self.child_segment)

        # Thorax is global check
        if self.parent_segment == Segment.THORAX:
            if self.extract_is_thorax_global(self.parent_segment):
                parent_is_thorax_global = True
                if parent_is_correctable is True:
                    parent_output = self._check_segment_has_to_isb_like_correction(
                        parent_correction, print_warnings=print_warnings
                    )
                elif parent_is_correctable is False:
                    parent_output = self._check_segment_has_no_correction(
                        parent_correction, print_warnings=print_warnings
                    )
                else:
                    print(
                        "The correction of thorax should be filled with a boolean value, "
                        "as it is a global coordinate system."
                    )

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
            parent_output = self._check_segment_has_no_correction(parent_correction, print_warnings=print_warnings)
            self.parent_segment_usable_for_rotation_data = parent_output
            self.parent_segment_usable_for_translation_data = False

        if self.child_biomech_sys.is_isb_oriented and self.child_biomech_sys.is_origin_on_an_isb_axis():
            child_output = self._check_segment_has_no_correction(child_correction, print_warnings=print_warnings)
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

        self.translation_isb_matrix_callback = (
            lambda trans_x, trans_y, trans_z: self.child_biomech_sys.get_rotation_matrix()
            @ np.array([[trans_x, trans_y, trans_z]]).T
        )

        if self.left_side:
            self.translation_mediolateral_matrix = (
                lambda trans_x, trans_y, trans_z: self.translation_isb_matrix_callback(trans_x, trans_y, trans_z)
                * np.array([1, 1, -1])
            )
        else:
            self.translation_mediolateral_matrix = self.translation_isb_matrix_callback

    def compute_deviations(self):
        """
        Compute the deviation of the joint from the ISB recommendation.
        """
        if self.has_rotation_data:
            rotation_parent_deviation = SegmentCompliance(mode="rotation", bsys=self.parent_biomech_sys)
            rotation_child_deviation = SegmentCompliance(mode="rotation", bsys=self.child_biomech_sys)
            rotation_joint_deviation = JointCompliance(
                mode="rotation", joint=self.joint, thoracohumeral_angle=self.thoracohumeral_angle
            )

            self.rotation_deviation = [rotation_parent_deviation, rotation_child_deviation, rotation_joint_deviation]

        if self.has_translation_data:
            translation_parent_deviation = SegmentCompliance(mode="translation", bsys=self.parent_biomech_sys)
            translation_child_deviation = SegmentCompliance(mode="translation", bsys=self.child_biomech_sys)
            translation_joint_deviation = JointCompliance(
                mode="translation", joint=self.joint, thoracohumeral_angle=self.thoracohumeral_angle
            )

            self.translation_deviation = [
                translation_parent_deviation,
                translation_child_deviation,
                translation_joint_deviation,
            ]

    @property
    def enough_compliant_for_translation(self) -> bool:
        """Check if the segment is compliant enough for merging translation data"""
        parent_deviation = SegmentCompliance(mode="translation", bsys=self.parent_biomech_sys)
        child_deviation = SegmentCompliance(mode="translation", bsys=self.child_biomech_sys)

        thoracohumeral_angle = set_thoracohumeral_angle_from_row(self.row)
        joint_deviation = JointCompliance(mode="rotation", joint=self.joint, thoracohumeral_angle=thoracohumeral_angle)

        pc1 = not parent_deviation.is_c1
        pc2 = not parent_deviation.is_c2
        pc3 = not parent_deviation.is_c3
        cc3 = not child_deviation.is_c3
        c5 = not joint_deviation.is_c5

        any_origin_is_wrong = not pc3 or not cc3

        if any_origin_is_wrong:
            return False

        all_good = pc1 and pc2 and pc3 and cc3
        orientation_correction = pc2 and pc3 and cc3  # flipping axis or changing the sign of the axis
        accepted_orientation_offset = pc1 and pc3 and cc3

        segment_conditions = all_good or orientation_correction or accepted_orientation_offset

        if segment_conditions and c5:
            return True

        if segment_conditions and not c5:
            # todo: if enough data, we can recompute the rotation matrices and convert the translations into another coordinate system
            #   e.g. joint coordinate system (euler basis) to proximal segment coordinate system for Moissenet et al.
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

        self.translation_data["article"] = self.row.shoulder_id
        self.translation_data["joint"] = JointType.from_string(self.row.joint)
        self.translation_data["humeral_motion"] = self.row.humeral_motion

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
        # to_concat_1dof = [
        #     pd.DataFrame(
        #         columns=[
        #             "unit",
        #             "humerothoracic_angle",
        #             "value",
        #             "degree_of_freedom",
        #             "article",
        #             "joint",
        #             "humeral_motion",
        #             "shoulder_id",
        #             "in_vivo",
        #             "xp_mean",
        #             "biomechanical_dof",
        #         ]
        #     )
        # ]
        prefix = f"{"corrected_" if correction else ""}df"

        if rotation:
            self.to_series_dataframe(correction=correction, rotation=True)
            # to_concat_1dof.append(getattr(self, f"{prefix}_rotation_1dof_per_line"))
            to_concat_3dof.append(getattr(self, f"{prefix}_rotation_3dof_per_line"))

        if translation:
            self.to_series_dataframe(correction=correction, rotation=False)
            # to_concat_1dof.append(getattr(self, f"{prefix}_translation_1dof_per_line"))
            to_concat_3dof.append(getattr(self, f"{prefix}_translation_3dof_per_line"))

        setattr(
            self,
            f"{prefix}_3dof_per_line",
            pd.concat(
                to_concat_3dof if len(to_concat_3dof) >= 1 else [get_empty_series_dataframe()],
            ),
        )
        # setattr(
        #     self,
        #     f"{prefix}_1dof_per_line",
        #     pd.concat(
        #         to_concat_1dof,
        #     ),
        # )
        # return self.corrected_df_1dof_per_line if correction else self.df_1dof_per_line
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
            # setattr(
            #     self,
            #     f"{prefix}_1dof_per_line",
            #     pd.DataFrame(
            #         columns=[
            #             "unit",
            #             "humerothoracic_angle",
            #             "value",
            #             "degree_of_freedom",
            #             "article",
            #             "joint",
            #             "humeral_motion",
            #             "shoulder_id",
            #             "in_vivo",
            #             "xp_mean",
            #             "biomechanical_dof",
            #         ]
            #     ),
            # )
            return series_dataframe

        no_correction_legend = ("x", "y", "z") if not rotation else tuple(self.joint.euler_sequence.value)
        correction_legend = ("x", "y", "z") if not rotation else self.joint.isb_rotation_biomechanical_dof
        three_dof_legend = correction_legend if correction else no_correction_legend
        value_dof = self.calculate_dof_values(
            data,
            correction_callable=self.apply_correction_in_radians if rotation else self.apply_correction_to_translation,
        )

        series_dataframe["value_dof1"] = value_dof[:, 0]
        series_dataframe["value_dof2"] = value_dof[:, 1]
        series_dataframe["value_dof3"] = value_dof[:, 2]
        series_dataframe["legend_dof1"] = three_dof_legend[0]
        series_dataframe["legend_dof2"] = three_dof_legend[1]
        series_dataframe["legend_dof3"] = three_dof_legend[2]
        series_dataframe["humerothoracic_angle"] = data["humerothoracic_angle"]

        series_dataframe["unit"] = "rad" if rotation else "mm"

        series_dataframe["article"] = self.row.dataset_authors
        series_dataframe["joint"] = self.row.joint
        series_dataframe["humeral_motion"] = self.row.humeral_motion

        series_dataframe["shoulder_id"] = self.row.shoulder_id
        series_dataframe["in_vivo"] = self.row.in_vivo
        series_dataframe["xp_mean"] = self.row.experimental_mean

        setattr(self, f"{prefix}_3dof_per_line", series_dataframe)
        # setattr(self, f"{prefix}_1dof_per_line", convert_df_to_1dof_per_line(series_dataframe, three_dof_legend))

        # return getattr(self, f"{prefix}_1dof_per_line")
        return getattr(self, f"{prefix}_3dof_per_line")

    @staticmethod
    def calculate_dof_values(data: pd.DataFrame, correction_callable: callable = None):
        """
        Calculate the dof values, assign values of correction if needed

        Parameters
        ----------
        data : pd.DataFrame
            The data to calculate the dof values
        correction_callable : callable, optional
            The callable to apply the correction, by default None, thus no correction applied
        """
        value_dof = np.zeros((data.shape[0], 3))

        if correction_callable is not None:
            for i, row in enumerate(data.itertuples()):
                corrected_dof_1, corrected_dof_2, corrected_dof_3 = correction_callable(
                    row.value_dof1, row.value_dof2, row.value_dof3
                )
                value_dof[i, 0] = corrected_dof_1
                value_dof[i, 1] = corrected_dof_2
                value_dof[i, 2] = corrected_dof_3

            # unwrap the angles to avoid discontinuities between -180 and 180 for example
            for i in range(0, 3):
                value_dof[:, i] = np.unwrap(value_dof[:, i], period=180)
        else:
            value_dof[:, 0] = data["value_dof1"].values
            value_dof[:, 1] = data["value_dof2"].values
            value_dof[:, 2] = data["value_dof3"].values

        return value_dof

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

    def apply_correction_to_translation(self, dof1, dof2, dof3) -> tuple[float, float, float]:
        """Apply the correction to the translation in mm, as we use a matrix product, we need nan to be zeros"""

        dof1 = dof1 if not np.isnan(dof1) else 0
        dof2 = dof2 if not np.isnan(dof2) else 0
        dof3 = dof3 if not np.isnan(dof3) else 0

        corrected_dof_1, corrected_dof_2, corrected_dof_3 = self.translation_mediolateral_matrix(dof1, dof2, dof3)

        deg_corrected_dof_1 = corrected_dof_1 if corrected_dof_1 != 0 else np.nan
        deg_corrected_dof_2 = corrected_dof_2 if corrected_dof_2 != 0 else np.nan
        deg_corrected_dof_3 = corrected_dof_3 if corrected_dof_3 != 0 else np.nan

        return deg_corrected_dof_1, deg_corrected_dof_2, deg_corrected_dof_3


def get_empty_series_dataframe():
    return pd.DataFrame(
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
            # "confidence",  # float
            "shoulder_id",  # int
            "in_vivo",  # bool
            "xp_mean",  # string
        ],
    )
