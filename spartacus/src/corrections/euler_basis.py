from numpy import ndarray
import numpy as np
from biorbd import Rotation

from ..enums_biomech import CartesianAxis, EulerSequence
from .angle_conversion_callbacks import from_euler_angles_to_rotation_matrix


def rotation_matrix_from_numpy_to_biorbd(R: np.ndarray) -> Rotation:
    """
    This function returns the rotation matrix in biorbd formalism

    Parameters
    ---------
    R : np.ndarray
        Rotation matrix (3x3)

    Returns
    ---------
    biorbd.Rotation
        The rotation matrix object
    """

    return Rotation(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )


def rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """
    This function returns the rotation matrix in euler angles vector

    Parameters
    ---------
    rotation_matrix : np.ndarray
        Rotation matrix (3x3)
    seq: str = "xyz"
        order of the coordinates in the returned vector
    Returns
    ---------
    Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_array()
        The Euler vector in radian as an array
    """

    rotation_matrix_biorbd = rotation_matrix_from_numpy_to_biorbd(rotation_matrix)
    return Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_array()


def rotation_x(angle: float) -> ndarray:
    """This function returns the rotation matrix around the x-axis for a given angle in radians."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, cos_angle, -sin_angle], [0.0, sin_angle, cos_angle]])


def rotation_y(angle: float) -> ndarray:
    """This function returns the rotation matrix around the y-axis for a given angle in radians."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, 0.0, sin_angle], [0.0, 1.0, 0.0], [-sin_angle, 0.0, cos_angle]])


def rotation_z(angle: float) -> ndarray:
    """This function returns the rotation matrix around the z-axis for a given angle in radians."""
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[cos_angle, -sin_angle, 0.0], [sin_angle, cos_angle, 0.0], [0.0, 0.0, 1.0]])


def rotation_matrices_from_rotation_matrix(rotation_matrix, sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function converts a rotation matrix to individual rotation matrices

    Parameters
    ----------
    rotation_matrix : np.ndarray
        Rotation matrix
    sequence : str
        Sequence of rotations, e.g. 'xyz'

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Individual rotation matrices
    """

    angles = rotation_matrix_to_euler_angles(rotation_matrix, sequence)

    R0 = rotation_matrix_from_angle_and_axis(angles[0], sequence[0])
    R1 = rotation_matrix_from_angle_and_axis(angles[1], sequence[1])
    R2 = rotation_matrix_from_angle_and_axis(angles[2], sequence[2])

    return R0, R1, R2


def rotation_matrix_from_angle_and_axis(angle: float, axis: str | CartesianAxis) -> np.ndarray:
    """
    This function returns a rotation matrix from an angle and an axis

    Parameters
    ----------
    angle : float
        Angle of rotation
    axis : str or CartesianAxis
        Axis of rotation such as 'x', 'y' or 'z'

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    axis_map = {
        "x": rotation_x(angle),
        CartesianAxis.plusX: rotation_x(angle),
        "y": rotation_y(angle),
        CartesianAxis.plusY: rotation_y(angle),
        "z": rotation_z(angle),
        CartesianAxis.plusZ: rotation_z(angle),
    }

    output = axis_map.get(axis)
    if output is None:
        raise ValueError("The axis must be 'x', 'y' or 'z'.")
    return output


def euler_axes_from_rotation_matrices(
    R_0_parent: np.ndarray,
    R_0_child: np.ndarray,
    sequence: EulerSequence,
    axes_source_frame: str = "mixed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function returns the euler axes from the rotation matrices in the global frame

    Parameters
    ----------
    R_0_parent : np.ndarray
        Rotation matrix of the parent ^0R_parent
    R_0_child : np.ndarray
        Rotation matrix of the child ^0R_child
    sequence : EulerSequence
        Sequence of rotations, e.g. 'xyz'
    axes_source_frame : str
        Frame from which the axes get computed from, e.g. 'parent', 'child', 'mixed'.
            - parent: the axes are computed from the parent rotation matrix
            - child: the axes are computed from the child rotation matrix
            - mixed (default): the first and second axes are computed from the parent rotation matrix,
            the third from the child, limitating non-linearities

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Euler rotation axes in the global frame
    """

    R = R_0_parent.T @ R_0_child
    if isinstance(sequence, EulerSequence):
        sequence = sequence.value

    individual_rotation_matrices = rotation_matrices_from_rotation_matrix(R, sequence)

    euler_axes = [None for _ in range(3)]

    cumulated_rotation_matrix = np.eye(3)

    if axes_source_frame == "parent":
        for i, (axis, rotation) in enumerate(zip(sequence, individual_rotation_matrices)):
            cumulated_rotation_matrix = cumulated_rotation_matrix @ rotation

            v = vector_from_axis(axis)
            # e.g. for xyz, x: R_0_parent @ rotx @ [1, 0, 0]; y: R_0_parent @ rotx @ roty @ [0, 1, 0]; ...
            euler_axes[i] = R_0_parent @ cumulated_rotation_matrix @ v

        return tuple(euler_axes)

    elif axes_source_frame == "child":
        # flip the sequence and the rotation matrices
        sequence = sequence[::-1]
        individual_rotation_matrices = individual_rotation_matrices[::-1]

        for i, (axis, rotation) in enumerate(zip(sequence, individual_rotation_matrices)):
            cumulated_rotation_matrix = cumulated_rotation_matrix @ rotation.T

            v = vector_from_axis(axis)
            # e.g. for xyz, z: R_0_child @ rotz.T @ [0, 0, 1]; y: R_0_child @ rotz.T @ roty.T @ [0, 1, 0]; ...
            euler_axes[i] = R_0_child @ cumulated_rotation_matrix @ v

        # flip the euler axes
        euler_axes = euler_axes[::-1]
        return tuple(euler_axes)

    elif axes_source_frame == "mixed":
        # this method should be better as it relies less on the transformations of the rotation matrices
        # only the second axis depends on the first angle, the third relies on the child matrix

        parent_euler_axes = euler_axes_from_rotation_matrices(
            R_0_parent, R_0_child, sequence, axes_source_frame="parent"
        )
        child_euler_axes = euler_axes_from_rotation_matrices(R_0_parent, R_0_child, sequence, axes_source_frame="child")

        return parent_euler_axes[0], parent_euler_axes[1], child_euler_axes[2]


def vector_from_axis(axis: str | CartesianAxis) -> np.ndarray:
    """
    This function returns the vector associated with an axis

    Parameters
    ----------
    axis : str or CartesianAxis
        Axis of rotation such as 'x', 'y' or 'z'

    Returns
    -------
    np.ndarray
        Vector associated with the axis
    """
    axis_map = {
        "x": np.array([1, 0, 0]),
        CartesianAxis.plusX: np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        CartesianAxis.plusY: np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
        CartesianAxis.plusZ: np.array([0, 0, 1]),
    }

    output = axis_map.get(axis)
    if output is None:
        raise ValueError("The axis must be 'x', 'y' or 'z'.")
    else:
        return output


def euler_angles_from_rotation_matrix(
    parent_matrix: np.ndarray, child_matrix: np.ndarray, joint_sequence: EulerSequence
) -> np.ndarray:
    """
    This function returns the euler angles from two rotation matrices

    Parameters
    ----------
    parent_matrix: np.ndarray
        The parent rotation matrix
    child_matrix: np.ndarray
        The child rotation matrix
    joint_sequence: EulerSequence
        The sequence of rotations

    Returns
    -------
    np.ndarray
        The euler angles
    """

    rot = parent_matrix.T @ child_matrix
    euler_angles = rotation_matrix_to_euler_angles(rot, joint_sequence.value)

    return euler_angles


def from_jcs_to_parent_frame(
    vector: np.ndarray,
    euler_angles: np.ndarray,
    sequence: EulerSequence,
) -> np.ndarray:
    """
    This function returns the joint rotation matrix in the parent frame
    from Euler basis frame

    Parameters
    ----------
    vector: np.ndarray
        The translations
    euler_angles: np.ndarray
        The rotations, Euler angles
    sequence: EulerSequence
        The sequence of rotations

    Returns
    -------
    np.ndarray
        The vector in the parent frame instead of the Euler basis frame
    """
    R_0_parent = np.eye(3)
    R_0_child = from_euler_angles_to_rotation_matrix(sequence.value, euler_angles[0], euler_angles[1], euler_angles[2])
    euler_axes = euler_axes_from_rotation_matrices(
        R_0_parent=R_0_parent,
        R_0_child=R_0_child,
        sequence=sequence,
        axes_source_frame="mixed",
    )
    R_euler_proximal = np.vstack(euler_axes)
    return np.linalg.inv(R_euler_proximal) @ vector
