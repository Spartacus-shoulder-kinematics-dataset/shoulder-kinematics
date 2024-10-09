import biorbd
import numpy as np
import pandas as pd

from .enums_biomech import Segment, JointType, EulerSequence
from .constants import REPEATED_DATAFRAME_KEYS


def mat_2_rotation(R: np.ndarray) -> biorbd.Rotation:
    """Convert a 3x3 matrix to a biorbd.Rotation"""
    return biorbd.Rotation(R[0, 0], R[0, 1], R[0, 2], R[1, 0], R[1, 1], R[1, 2], R[2, 0], R[2, 1], R[2, 2])


def flip_rotations(angles: np.ndarray, seq: str) -> np.ndarray:
    """
    Return an alternate sequence with the second angle inverted, but that
        leads to the same rotation matrices. See below for more information.

    Parameters
    ----------
    angles: np.ndarray
        The rotation angles
    seq: str
        The sequence of the rotation angles

    Returns
    -------
    np.ndarray
        The rotation angles flipped


    Source
    ------
    github.com/felixchenier/kineticstoolkit/blob/24e3dd39a6546d475732b70609c07fcc26dc2ff7/kineticstoolkit/geometry.py#L526-L537

    Notes
    -----
    Before flipping, the angles are:

    - First angle belongs to [-180, 180] degrees (both inclusive)
    - Second angle belongs to:

        - [-90, 90] degrees if all axes are different. e.g., xyz
        - [0, 180] degrees if first and third axes are the same e.g., zxz

    - Third angle belongs to [-180, 180] degrees (both inclusive)

    If after flipping, the angles are:

    - First angle belongs to [-180, 180] degrees (both inclusive)
    - Second angle belongs to:

        - [-180, -90], [90, 180] degrees if all axes are different. e.g., xyz
        - [-180, 0] degrees if first and third axes are the same e.g., zxz

    - Third angle belongs to [-180, 180] degrees (both inclusive)
    """
    offset = np.pi  # only in radians

    if seq[0] == seq[2]:  # Euler angles
        angles[0] = np.mod(angles[0], 2 * offset) - offset
        angles[1] = -angles[1]
        angles[2] = np.mod(angles[2], 2 * offset) - offset
    else:  # Tait-Bryan angles
        angles[0] = np.mod(angles[0], 2 * offset) - offset
        angles[1] = offset - angles[1]
        angles[angles[1] > offset, :] -= 2 * offset
        angles[2] = np.mod(angles[2], 2 * offset) - offset

    return angles


def get_segment_columns(segment: Segment) -> list[str]:
    columns = {
        Segment.THORAX: ["thorax_x", "thorax_y", "thorax_z", "thorax_origin"],
        Segment.CLAVICLE: ["clavicle_x", "clavicle_y", "clavicle_z", "clavicle_origin"],
        Segment.SCAPULA: ["scapula_x", "scapula_y", "scapula_z", "scapula_origin"],
        Segment.HUMERUS: ["humerus_x", "humerus_y", "humerus_z", "humerus_origin"],
    }

    the_columns = columns.get(segment, ValueError(f"{segment} is not a valid segment."))
    add_suffix = "_sense"
    return [f"{column}{add_suffix}" for column in the_columns[:3]] + [the_columns[3]]


def get_segment_columns_direction(segment: Segment) -> list[str]:
    columns = {
        Segment.THORAX: ["thorax_x", "thorax_y", "thorax_z", "thorax_origin"],
        Segment.CLAVICLE: ["clavicle_x", "clavicle_y", "clavicle_z", "clavicle_origin"],
        Segment.SCAPULA: ["scapula_x", "scapula_y", "scapula_z", "scapula_origin"],
        Segment.HUMERUS: ["humerus_x", "humerus_y", "humerus_z", "humerus_origin"],
    }

    the_columns = columns.get(segment, ValueError(f"{segment} is not a valid segment."))
    add_suffix = "_direction"
    return [f"{column}{add_suffix}" for column in the_columns[:3]] + [the_columns[3]]


def get_is_isb_column(segment: Segment) -> str:
    columns = {
        Segment.THORAX: "thorax_is_isb",
        Segment.CLAVICLE: "clavicle_is_isb",
        Segment.SCAPULA: "scapula_is_isb",
        Segment.HUMERUS: "humerus_is_isb",
    }
    return columns.get(segment, ValueError(f"{segment} is not a valid segment."))


def get_correction_column(segment: Segment) -> str:
    columns = {
        Segment.THORAX: "thorax_correction_method",
        Segment.CLAVICLE: "clavicle_correction_method",
        Segment.SCAPULA: "scapula_correction_method",
        Segment.HUMERUS: "humerus_correction_method",
    }
    return columns.get(segment, ValueError(f"{segment} is not a valid segment."))


def compute_rotation_matrix_from_axes(
    anterior_posterior_axis: np.ndarray,
    infero_superior_axis: np.ndarray,
    medio_lateral_axis: np.ndarray,
):
    """
    Compute the rotation matrix from the axes of the ISB coordinate system, the rotation matrix from the axes,
    named R_isb_local such that v_isb = R_isb_local @ v_local

    Parameters
    ----------
    anterior_posterior_axis: np.ndarray
        The anterior-posterior axis expressed in the ISB coordinate system
    infero_superior_axis: np.ndarray
        The infero-superior axis expressed in the ISB coordinate system
    medio_lateral_axis: np.ndarray
        The medio-lateral axis expressed in the ISB coordinate system

    Returns
    -------
    np.ndarray
        The rotation matrix from the ISB coordinate system to the local coordinate system
        R_isb_local
        meaning when a vector v expressed in local coordinates is transformed to ISB coordinates
        v_isb = R_isb_local @ v_local
    """
    return np.array(
        [
            # X axis                                    Y axis                                      Z axis ,
            #  in ISB base
            [
                anterior_posterior_axis[0, 0],
                infero_superior_axis[0, 0],
                medio_lateral_axis[0, 0],
            ],
            [
                anterior_posterior_axis[1, 0],
                infero_superior_axis[1, 0],
                medio_lateral_axis[1, 0],
            ],
            [
                anterior_posterior_axis[2, 0],
                infero_superior_axis[2, 0],
                medio_lateral_axis[2, 0],
            ],
        ],
        dtype=np.float64,
    ).T  # where the transpose was missing in the original code


def convert_df_to_1dof_per_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a dataframe with 3 degrees of freedom per line to a dataframe with 1 degree of freedom per line
    stacking dof 1 then dof2 and then dof 3 under each others
    """

    first_dict = {key: df[key].values[:, np.newaxis].repeat(3, axis=1).T.flatten() for key in REPEATED_DATAFRAME_KEYS}
    second_dict = {
        "humerothoracic_angle": df["humerothoracic_angle"].values[:, np.newaxis].repeat(3, axis=1).T.flatten(),
        "value": df[["value_dof1", "value_dof2", "value_dof3"]].values.T.flatten(),
        "legend": df[["legend_dof1", "legend_dof2", "legend_dof3"]].values.T.flatten(),
        "degree_of_freedom": np.array([[1, 2, 3]]).repeat(repeats=df.shape[0], axis=0).T.flatten(),
    }
    df_transformed = pd.DataFrame({**first_dict, **second_dict})

    return df_transformed


def calculate_dof_values(
    data: pd.DataFrame,
    correction_callable: callable = None,
    rotation: bool = None,
    rotation_data: pd.DataFrame = None,
) -> np.ndarray:
    """
    Calculate the dof values, assign values of correction if needed

    Parameters
    ----------
    data : pd.DataFrame
        The data to calculate the dof values
    correction_callable : callable, optional
        The callable to apply the correction, by default None, thus no correction applied
    rotation : bool, optional
        If True, apply the correction on rotation data, by default None
    rotation_data : pd.DataFrame, optional
        The rotation data to use for correction, by default None
    """
    value_dof = np.zeros((data.shape[0], 3))

    if correction_callable is not None and rotation:
        for i, row in enumerate(data.itertuples()):
            corrected_dof_1, corrected_dof_2, corrected_dof_3 = correction_callable(
                row.value_dof1, row.value_dof2, row.value_dof3
            )
            value_dof[i, 0] = corrected_dof_1
            value_dof[i, 1] = corrected_dof_2
            value_dof[i, 2] = corrected_dof_3

        mvt = data["humeral_motion"].unique()[0]
        joint = data["joint"].unique()[0]
        if joint == JointType.GLENO_HUMERAL and mvt in ("internal-external rotation 0 degree-abducted",):
            for i in range(0, 3):
                value_dof[:, i] = np.unwrap(value_dof[:, i], period=180)

        return value_dof

    elif correction_callable is not None and not rotation:

        for i, row in enumerate(data.itertuples()):
            # get the rotation data for the corresponding translation values for further correction if needed (jcs to proximal)
            translation_th_angle = row.humerothoracic_angle
            subrotation_data = rotation_data[rotation_data["humerothoracic_angle"] == translation_th_angle]
            subrotation_data = subrotation_data[subrotation_data["unit"] == "rad"]
            if rotation_data.empty:
                rot1, rot2, rot3 = None, None, None
            else:
                # todo: TO REMOVE !!
                try:
                    rot1 = subrotation_data["value_dof1"].values[0]
                    rot2 = subrotation_data["value_dof2"].values[0]
                    rot3 = subrotation_data["value_dof3"].values[0]
                except:
                    rot1, rot2, rot3, seq = None, None, None, None

            corrected_dof_1, corrected_dof_2, corrected_dof_3 = correction_callable(
                row.value_dof1, row.value_dof2, row.value_dof3, rot1, rot2, rot3
            )
            value_dof[i, 0] = corrected_dof_1
            value_dof[i, 1] = corrected_dof_2
            value_dof[i, 2] = corrected_dof_3

        mvt = data["humeral_motion"].unique()[0]
        joint = data["joint"].unique()[0]
        if joint == JointType.GLENO_HUMERAL and mvt in ("internal-external rotation 0 degree-abducted",):
            for i in range(0, 3):
                value_dof[:, i] = np.unwrap(value_dof[:, i], period=180)

        return value_dof

    else:

        value_dof[:, 0] = data["value_dof1"].values
        value_dof[:, 1] = data["value_dof2"].values
        value_dof[:, 2] = data["value_dof3"].values
        return value_dof
