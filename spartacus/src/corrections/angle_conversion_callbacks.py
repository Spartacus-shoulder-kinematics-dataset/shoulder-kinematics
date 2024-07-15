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

    return rotation_matrix_2_euler_angles(isb_framed_rotation_matrix, EulerSequence.from_string(new_sequence_str))


def quick_fix_x_rot_in_yxy(new_angles: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    🔄 Quick Fix for X Rotation in YXY Euler Angle Sequence: Resolving the β Ambiguity 🧭

    This function addresses a critical issue in matrix-to-Euler conversions for YXY sequences,
    specifically focusing on resolving the ambiguity of the x-axis rotation (β).

    Key Concepts:
    1. YXY Euler Angle Sequence: R = Ry(α) * Rx(β) * Ry(γ)
       where α, γ ∈ [-π, π] and traditionally β ∈ [0, π]

    R = [cos(α)cos(γ) - sin(α)cos(β)sin(γ)    sin(α)sin(β)    cos(α)sin(γ) + sin(α)cos(β)cos(γ)]
        [sin(β)sin(γ)                         cos(β)          -sin(β)cos(γ)                    ]
        [-sin(α)cos(γ) - cos(α)cos(β)sin(γ)   cos(α)sin(β)    -sin(α)sin(γ) + cos(α)cos(β)cos(γ)]

    2. The Fundamental Issue:
       Standard conversion (β = arccos(r22)) fails to distinguish between β and -β,
       as cos(-β) = cos(β).

    3. The Ambiguity:
       (α, β, γ) and (α ± π, -β, γ ± π) can represent the same rotation.

    4. Our Solution:
       We introduce a check based on sin(β):
       If matrix[1, 0] < 0 or matrix[1, 2] > 0, we infer β < 0 and adjust accordingly.

    Parameters:
    new_angles (np.ndarray): Array of Euler angles [α, β, γ] in radians,
                             where β is the rotation around the x-axis.
    matrix (np.ndarray): 3x3 rotation matrix corresponding to the YXY sequence.

    Returns:
    np.ndarray: Corrected Euler angles with the proper sign for the x rotation (β).

    Note:
    This function resolves the β sign ambiguity by checking matrix[1, 0] (sin(β)sin(γ))
    and matrix[1, 2] (-sin(β)cos(γ)). It works for all values of γ and ensures correct
    sign determination even with floating-point precision issues near extreme angles.

    Remember: In the realm of 3D rotations, not all paths lead to Rome,
    but they might lead to the same orientation! 🌐
    """

    if matrix[1, 0] < 0 or matrix[1, 2] > 0:
        new_angles[1] *= -1
        new_angles[0] += np.pi
        new_angles[2] += np.pi

    return new_angles


def rotation_matrix_2_euler_angles(
    rotation_matrix: np.ndarray,
    euler_sequence: EulerSequence,
) -> np.ndarray:
    rotation_matrix_object = mat_2_rotation(rotation_matrix)
    new_angles = biorbd.Rotation.toEulerAngles(rotation_matrix_object, seq=euler_sequence.value.lower()).to_array()

    return quick_fix_x_rot_in_yxy(new_angles, rotation_matrix) if euler_sequence == EulerSequence.YXY else new_angles


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
