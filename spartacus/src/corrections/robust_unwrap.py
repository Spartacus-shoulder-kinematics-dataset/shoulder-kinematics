import numpy as np
import biorbd
from scipy import optimize
from typing import Literal


def helicoidal_angle(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the helicoidal angle from a rotation matrix.

    Parameters:
    rotation_matrix (np.ndarray): A 3x3 rotation matrix.

    Returns:
    np.ndarray: The helicoidal angle in radians.
    """
    return np.arccos((np.trace(rotation_matrix) - 1) / 2)


def unwrap_rotation_matrix_from_euler_angles(angles, seq: str, angles_init: np.ndarray):
    """
    Unwrap Euler angles based on the first rotation matrix.

    Parameters:
    angles (np.ndarray): The Euler angles to unwrap.
    seq (str): The sequence of rotations (e.g., 'xyz', 'zyx').
    angles_init (np.ndarray): Initial guess for the unwrapped angles.

    Returns:
    np.ndarray: The unwrapped Euler angles.
    """
    R = biorbd.Rotation.fromEulerAngles(rot=angles, seq=seq).to_array()
    return unwrap_rotation_matrix_from_matrix(R, seq, angles_init)


def unwrap_rotation_matrix_from_matrix(rotation_matrix: np.ndarray, seq: str, angles_init: np.ndarray):
    """
    Unwrap a rotation matrix to Euler angles.

    Parameters:
    rotation_matrix (np.ndarray): The rotation matrix to unwrap.
    seq (str): The sequence of rotations (e.g., 'xyz', 'zyx').
    angles_init (np.ndarray): Initial guess for the unwrapped angles.

    Returns:
    np.ndarray: The unwrapped Euler angles.
    """
    objective_function = (
        lambda x: helicoidal_angle(biorbd.Rotation.fromEulerAngles(rot=x, seq=seq).to_array().T @ rotation_matrix)
        * 180
        / np.pi
    )
    sol = optimize.least_squares(fun=objective_function, x0=angles_init, verbose=0, method="trf")

    return sol.x


def unwrap_angles_from_euler_angles(
    angles_series: np.ndarray,
    seq: Literal["xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "xyx", "xzx", "yxy", "yzy", "zxz", "zyz"],
    angles_init: np.ndarray = None,
    init_strategy: Literal["fixed", "previous"] = "previous",
) -> np.ndarray:
    """
    Unwrap a series of Euler angles to ensure continuity.

    This function processes a series of Euler angles and attempts to remove discontinuities
    by unwrapping the angles based on an initial guess and a specified sequence.

    Parameters:
    -----------
    angles_series : np.ndarray
        A series of Euler angles to unwrap, shape (n, 3) where n is the number of angle sets.
    seq : str
        The sequence of rotations (e.g., 'xyz', 'zyx', 'yxy'). Must be one of the 12 valid
        Euler angle sequences.
    angles_init : np.ndarray, optional
        Initial guess for the first set of unwrapped angles, shape (3,). The first element of angles_series by default
    init_strategy : {'fixed', 'previous'}, optional
        Strategy for choosing the initial guess for each iteration:
        - 'fixed': Always use angles_init (default)
        - 'previous': Use the result of the previous iteration

    Returns:
    --------
    np.ndarray
        The unwrapped series of Euler angles, shape (n, 3).

    Raises:
    -------
    ValueError
        If angles_series is not a 2D array with 3 columns, or if angles_init is not a 1D array with 3 elements.
    """
    if angles_series.ndim != 2 or angles_series.shape[1] != 3:
        raise ValueError("angles_series must be a 2D array with 3 columns")
    if angles_init is None:
        angles_init = angles_series[0, :]
    if angles_init.shape != (3,):
        raise ValueError("angles_init must be a 1D array with 3 elements")
    if not init_strategy in ("fixed", "previous"):
        raise ValueError("Invalid init_strategy. Must be 'fixed', 'previous', or 'original'.")

    new_angles = np.empty_like(angles_series)

    for i, angles in enumerate(angles_series):
        new_angles[i, :] = unwrap_rotation_matrix_from_euler_angles(angles, seq, initial_guess)
        initial_guess = new_angles[i, :] if init_strategy == "previous" else angles_init

    return new_angles
