import numpy as np
import pandas as pd

from .biomech_system import BiomechCoordinateSystem
from .enums_biomech import JointType, Segment, AnatomicalLandmark, Correction
from .joint import Joint
from .utils import (
    get_is_isb_column,
    # get_is_correctable_column,
    get_segment_columns_direction,
)
from .utils_setters import set_parent_segment_from_row


def check_parent_child_joint(bjoint: Joint, row: pd.Series, print_warnings: bool = False):
    """
    This function checks if the parent and child segment are compatible with the joint type.

    Parameters
    ----------
    bjoint : Joint
        The joint to check.
    row : pandas.Series
        The row of the dataset to check.
    print_warnings : bool, optional
        If True, print warnings when inconsistencies are found. The default is False.

    Returns
    -------
    bool
        True if the parent and child segment are compatible with the joint type, False otherwise.
    """
    if not _check_parent_child_joint(bjoint.joint_type, parent_name=row.parent, child_name=row.child):
        if print_warnings:
            print("WARNING : inconsistency in the dataset")
            print(row.joint, row.article_author_year)
            print("detected :", bjoint.joint_type)
            print("expected :", row.parent, row.child)
        return False
    return True


def _check_parent_child_joint(joint_type: JointType, parent_name: str, child_name: str) -> bool:
    parent_segment = Segment.from_string(parent_name)
    child_segment = Segment.from_string(child_name)

    if joint_type == JointType.GLENO_HUMERAL:
        # Scapula is the parent segment of the humerus
        return parent_segment == Segment.SCAPULA and child_segment == Segment.HUMERUS
    elif joint_type == JointType.ACROMIO_CLAVICULAR:
        # Clavicle is the parent segment of the scapula
        return parent_segment == Segment.CLAVICLE and child_segment == Segment.SCAPULA
    elif joint_type == JointType.STERNO_CLAVICULAR:
        # Clavicle is the parent segment of the thorax
        return parent_segment == Segment.THORAX and child_segment == Segment.CLAVICLE
    elif joint_type == JointType.THORACO_HUMERAL:
        # Thorax is the parent segment of the humerus
        return parent_segment == Segment.THORAX and child_segment == Segment.HUMERUS
    elif joint_type == JointType.SCAPULO_THORACIC:
        # Thorax is the parent segment of the scapula
        return parent_segment == Segment.THORAX and child_segment == Segment.SCAPULA
    else:
        raise ValueError(f"{joint_type} is not a valid joint type. Got {joint_type}, {parent_name}, {child_name}")


def check_segment_filled_with_nan(row: pd.Series, segment: list, print_warnings: bool = False):
    """
    This function checks if the segment is not given and filled with NaN values.

    Parameters
    ----------
    row : pandas.Series
        The row of the dataset to check.
    segment : list
        The list of the columns of the segment to check. e.g. ["humerus_x", "humerus_y", "humerus_z"]
    print_warnings : bool, optional
        If True, print warnings when inconsistencies are found. The default is False.

    Returns
    -------
    bool
        True if the segment is filled with NaN values, False otherwise.
    """
    if row[segment[0]] is None or row[segment[1]] is None or row[segment[2]] is None:
        if print_warnings:
            print(segment, " is filled with nan")
        return True
    if isinstance(row[segment[0]], float) or isinstance(row[segment[1]], float) or isinstance(row[segment[2]], float):
        if np.isnan(row[segment[0]]) or np.isnan(row[segment[1]]) or np.isnan(row[segment[2]]):
            if print_warnings:
                print(segment, " is filled with nan")
            return True
    return False


# def check_is_isb_segment(row: pd.Series, bsys: BiomechCoordinateSystem, print_warnings: bool = False) -> bool:
#     """
#     This function checks if the segment is ISB oriented and if it is well specified in the dataset.
#
#     Parameters
#     ----------
#     bsys : BiomechCoordinateSystem
#         The biomechanical coordinate system to check.
#     row : pandas.Series
#         The row of the dataset to check.
#     print_warnings : bool, optional
#         If True, print warnings when inconsistencies are found. The default is False.
#
#     Returns
#     -------
#     bool
#         True if the segment is ISB oriented and if it is well specified in the dataset, False otherwise.
#
#     Notes
#     -----
#     This function does not check if the segment is ISB oriented, it only checks if the dataset is well specified.
#     Ex: a segment can be entirely ISB in the dataset, but if the correctable column is filled with True or False, then
#     it means it is an ISB like segment, but not exactly ISB. Because it can either be corrected or not.
#     if nan/None is given in the correctable column, then it means we don't have to apply any correction, so the segment
#     is a well-defined ISB segment.
#
#     """
#     is_isb = get_is_isb_column(bsys.segment)
#     is_correctable_col = get_is_correctable_column(bsys.segment)
#
#     if not bsys.is_isb() == row[is_isb] and np.isnan(row[is_correctable_col]):
#         # if expected and detected are different for isb, and the correctable is set to nan, then there is an inconsistency
#         # False means we know we cannot correct it, True means we know we can correct it
#         if print_warnings:
#             print("WARNING : inconsistency in the dataset")
#             print("-- ", row.article_author_year, " --")
#             print(bsys.segment)
#             print("detected ISB oriented:", bsys.is_isb_oriented())
#             print("detected ISB origin:", bsys.is_isb_origin(), bsys.origin)
#             print("detected ISB oriented + origin:", bsys.is_isb())
#             print("expected ISB:", row[is_isb])
#         return False
#
#     return True


def check_correction_methods(row: "RowData", bsys: BiomechCoordinateSystem, print_warnings: bool = False) -> bool:
    """
    This function checks if the correction method is in accordance with the segment type
    if the segment is a scapula, we can find correction methods
    if the segment is a humerus, clavicle, or thorax, we cannot find correction methods
    as it is already stated in the previous column if it's possible to get isb segment or isb like segment

    Parameters
    ----------
    bsys : BiomechCoordinateSystem
        The biomechanical coordinate system to check.
    row : pandas.Series
        The row of the dataset to check.
    print_warnings : bool, optional
        If True, print warnings when inconsistencies are found. The default is False.

    Returns
    -------
    bool
        True if the correction method is in accordance with the segment type, False otherwise.

    Notes
    -----

    """
    if bsys.segment == Segment.SCAPULA:
        # there is correction methods that can be applied to the scapula
        # even if it leads to isb like segment
        correction_cell = row.extract_corrections(bsys.segment)
        if correction_cell is None:
            return True
        else:
            if bsys.origin == AnatomicalLandmark.Scapula.GLENOID_CENTER:
                if Correction.SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION in correction_cell:
                    return True
                else:
                    if print_warnings:
                        print(
                            "WARNING : inconsistency in the dataset. "
                            "The correction method is not consistent with the segment origin."
                        )
                        print("-- ", row.row.article_author_year, " --")
                        print(bsys.origin)
                        print("detected correction method:", correction_cell)
                    return False
            if bsys.origin == AnatomicalLandmark.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER:
                if Correction.SCAPULA_KOLZ_AC_TO_PA_ROTATION in correction_cell:
                    return True
                else:
                    if print_warnings:
                        print(
                            "WARNING : inconsistency in the dataset. "
                            "The correction method is not consistent with the segment origin."
                        )
                        print("-- ", row.row.article_author_year, " --")
                        print(bsys.origin)
                        print("detected correction method:", correction_cell)
                    return False

    else:
        correction_cell = row.extract_corrections(bsys.segment)
        if correction_cell is not None:
            # there is a correction method
            # then is_correctable should be either true or false
            if print_warnings:
                print(
                    "WARNING : inconsistency in the dataset. "
                    "There should be no correction method for segment such as humerus, clavicle, or thorax."
                )
                print("-- ", row.article_author_year, " --")
                print(bsys.segment)
                print("detected correction method:", correction_cell)
            return False

        else:
            return True


def check_is_euler_sequence_provided(row: pd.Series, print_warnings: bool = False) -> bool:
    """This function checks if the euler sequence is provided in the dataset."""
    if row.euler_sequence is None:
        if print_warnings:
            print(
                "WARNING : euler sequence is not provided, for joint",
                row.joint,
                row.dataset_authors,
                f". Got {row.euler_sequence}",
            )
        return False
    # todo: check nan should disappear
    if not isinstance(row.euler_sequence, str) and (row.euler_sequence == "nan" or np.isnan(row.euler_sequence)):
        if print_warnings:
            print(
                "WARNING : euler sequence is nan, for joint",
                row.joint,
                row.dataset_authors,
                f". Got {row.euler_sequence}",
            )
        return False
    # check the three letters and remove "'" if present
    if not len(row.euler_sequence.replace("'", "")) == 3:
        if print_warnings:
            print(
                "WARNING : euler sequence is not 3 letters long, for joint",
                row.joint,
                row.dataset_authors,
                f". Got {row.euler_sequence}",
            )
        return False
    # check if the letters are x, y, or z
    authorized_letters = ["x", "y", "z"]
    if (
        not row.euler_sequence[0] in authorized_letters
        or not row.euler_sequence[1] in authorized_letters
        or not row.euler_sequence[3] in authorized_letters
    ):
        if print_warnings:
            print(
                "WARNING : euler sequence is not x, y, or z, for joint",
                row.joint,
                row.dataset_authors,
                f". Got {row.euler_sequence}",
            )
        return False

    return True


def check_same_orientation(
    parent: BiomechCoordinateSystem, child: BiomechCoordinateSystem, print_warnings: bool = False
) -> bool:
    """This function checks if the parent and child segments have the same orientation."""
    condition1 = parent.anterior_posterior_axis == child.anterior_posterior_axis
    condition2 = parent.medio_lateral_axis == child.medio_lateral_axis
    condition3 = parent.infero_superior_axis == child.infero_superior_axis

    output = True
    if not condition1 or not condition2 or not condition3:
        if print_warnings:
            print("WARNING : inconsistency in the dataset")
            print(
                "parent :",
                parent.segment,
                parent.anterior_posterior_axis,
                parent.medio_lateral_axis,
                parent.infero_superior_axis,
            )
            print(
                "child :",
                child.segment,
                child.anterior_posterior_axis,
                child.medio_lateral_axis,
                child.infero_superior_axis,
            )
        output = False

    return output


def check_is_translation_provided(row: pd.Series, print_warnings: bool = False) -> bool:
    """This function checks if the translation is provided in the dataset."""
    # check that the column origin_displacement and displacement_cs (coordinate system) are not nan

    origin_displacement_provided = isinstance(row.origin_displacement, str) and (
        not row.origin_displacement == "nan" or not np.isnan(row.origin_displacement)
    )
    displacement_cs_provided = isinstance(row.displacement_cs, str) and (
        not row.displacement_cs == "nan" or not np.isnan(row.displacement_cs)
    )

    if not origin_displacement_provided or not displacement_cs_provided:
        if print_warnings:
            print("WARNING : translation is not entirely provided, for joint", row.joint, row.dataset_authors)
            print(f"origin_displacement_provided : {origin_displacement_provided}")
            print(f"displacement_cs_provided : {displacement_cs_provided}")
        return False
    return True


def check_all_segments_validity(row, print_warnings: bool = False) -> bool:
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
        if check_segment_filled_with_nan(row, segment_cols_direction, print_warnings=print_warnings):
            continue

        bsys = set_parent_segment_from_row(row, segment_enum)

        # third check if the segment is direct or not
        if not bsys.is_direct():
            if print_warnings:
                print(f"{row.dataset_authors}, " f"Segment {segment_enum.value} is not direct, " f"it should be !!!")
            output = False

    return output
