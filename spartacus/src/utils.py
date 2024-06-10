import biorbd
import numpy as np

from .enums_biomech import Segment


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


def get_is_correctable_column(segment: Segment) -> str:
    columns = {
        Segment.THORAX: "thorax_is_isb_correctable",
        Segment.CLAVICLE: "clavicle_is_isb_correctable",
        Segment.SCAPULA: "scapula_is_isb_correctable",
        Segment.HUMERUS: "humerus_is_isb_correctable",
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
