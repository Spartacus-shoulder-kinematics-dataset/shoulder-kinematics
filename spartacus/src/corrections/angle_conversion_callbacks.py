import biorbd
import numpy as np

from ..biomech_system import BiomechCoordinateSystem
from ..enums_biomech import EulerSequence
from ..utils import mat_2_rotation


def get_angle_conversion_callback_from_tuple(tuple_factors: tuple[int, int, int]) -> callable:
    if not all([x in [-1, 1] for x in tuple_factors]):
        raise ValueError("tuple_factors must be a tuple of 1 and -1")

    return lambda rot1, rot2, rot3: (
        rot1 * tuple_factors[0],
        rot2 * tuple_factors[1],
        rot3 * tuple_factors[2],
    )


def convert_euler_angles(previous_sequence_str: str, new_sequence_str: str, rot1, rot2, rot3) -> np.ndarray:
    """Convert Euler angles from one sequence to another"""
    r = biorbd.Rotation.fromEulerAngles(np.array([rot1, rot2, rot3]), seq=previous_sequence_str)
    return biorbd.Rotation.toEulerAngles(r, seq=new_sequence_str).to_array()


def get_angle_conversion_callback_from_sequence(
    previous_sequence: EulerSequence, new_sequence: EulerSequence
) -> callable:
    # check if sequences are different
    if previous_sequence == new_sequence:
        raise ValueError("previous_sequence and new_sequence must be different")

    previous_sequence_str = previous_sequence.value.lower()
    new_sequence_str = new_sequence.value.lower()
    return lambda rot1, rot2, rot3: convert_euler_angles(previous_sequence_str, new_sequence_str, rot1, rot2, rot3)


def from_euler_angles_to_rotation_matrix(
    previous_sequence_str: str,
    rot1,
    rot2,
    rot3,
):
    rotation_matrix_object = biorbd.Rotation.fromEulerAngles(np.array([rot1, rot2, rot3]), seq=previous_sequence_str)
    rotation_matrix = rotation_matrix_object.to_array()
    return rotation_matrix


def isb_framed_rotation_matrix_from_euler_angles(
    previous_sequence_str: str,
    rot1,
    rot2,
    rot3,
    bsys_parent: BiomechCoordinateSystem,
    bsys_child: BiomechCoordinateSystem,
) -> np.ndarray:
    """
    Returns the joint rotation matrix in a ISB-like manner by recomputing the rotation matrix from previous sequence
    and applying rotation matrix to turn the parent and the child into ISB-like coordinate system

    Returns
    -------
    np.ndarray
        The joint rotation matrix in a ISB-like manner (we may add an extra correction later)
    """
    rotation_matrix = from_euler_angles_to_rotation_matrix(previous_sequence_str, rot1, rot2, rot3)
    converted_rotation_matrix = bsys_child.get_rotation_matrix() @ rotation_matrix @ bsys_parent.get_rotation_matrix().T

    return converted_rotation_matrix


def to_left_handed_frame(
    matrix: np.ndarray,
):
    """
    Convert a rotation matrix to a left-handed frame, by multiplying the z-axis by -1.

    Consequently, the determinant of the matrix will be -1.
    But, the identified euler angles of the left side (left shoulder) would have the same signs
    as for the right-handed frame of the right side (right shoulder).
    """
    return set_corrections_on_rotation_matrix(
        child_matrix_correction=np.diag([1, 1, -1]),
        matrix=matrix,
        parent_matrix_correction=np.diag([1, 1, -1]),
    )


def set_corrections_on_rotation_matrix(
    matrix: np.ndarray,
    child_matrix_correction: np.ndarray,
    parent_matrix_correction: np.ndarray,
):
    """Returns the rotation matrix with the child and parent correction applied"""
    return child_matrix_correction @ matrix @ parent_matrix_correction.T


def convert_euler_angles_and_frames_to_isb(
    previous_sequence_str: str,
    new_sequence_str: str,
    rot1,
    rot2,
    rot3,
    bsys_parent: BiomechCoordinateSystem,
    bsys_child: BiomechCoordinateSystem,
) -> np.ndarray:
    """
    Returns the Euler angles in ISB-like manner by recomputing the rotation matrix
    and applying rotation matrix to turn the parent and the child into ISB coordinate system
    """
    isb_framed_rotation_matrix = isb_framed_rotation_matrix_from_euler_angles(
        previous_sequence_str,
        rot1,
        rot2,
        rot3,
        bsys_parent,
        bsys_child,
    )

    new_rotation_matrix_object = mat_2_rotation(isb_framed_rotation_matrix)
    return biorbd.Rotation.toEulerAngles(new_rotation_matrix_object, seq=new_sequence_str).to_array()


def rotation_matrix_2_euler_angles(
    rotation_matrix: np.ndarray,
    euler_sequence: EulerSequence,
) -> np.ndarray:
    rotation_matrix_object = mat_2_rotation(rotation_matrix)
    return biorbd.Rotation.toEulerAngles(rotation_matrix_object, seq=euler_sequence.value.lower()).to_array()


def get_angle_conversion_callback_to_isb_with_sequence(
    previous_sequence: EulerSequence,
    new_sequence: EulerSequence,
    bsys_parent: BiomechCoordinateSystem,
    bsys_child: BiomechCoordinateSystem,
) -> callable:
    """
    Get the callback to convert euler angles from a sequence to ISB sequence
    by recomputing the rotation matrix,
    and applying rotation matrix to turn the parent and the child into ISB coordinate system
    """
    if previous_sequence == new_sequence:
        raise ValueError("previous_sequence and new_sequence must be different")

    previous_sequence_str = previous_sequence.value.lower()
    new_sequence_str = new_sequence.value.lower()

    return lambda rot1, rot2, rot3: convert_euler_angles_and_frames_to_isb(
        previous_sequence_str,
        new_sequence_str,
        rot1,
        rot2,
        rot3,
        bsys_parent,
        bsys_child,
    )
