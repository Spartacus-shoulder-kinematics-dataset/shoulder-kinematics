import numpy as np
from .robust_unwrap import unwrap_angles_from_euler_angles


def unwrap_for_yxy_glenohumeral_joint(three_dof_series: np.ndarray) -> np.ndarray:
    """
    Unwrap the angles in a series of 3-DOF angles to ensure continuity,
    particularly handling the glenohumeral joint's potential rotations for glenohumeral yxy sequence

    Parameters:
    three_dof_series (np.ndarray): A 2D numpy array with each row representing a 3-DOF angle (y, x, y).

    Returns:
    np.ndarray: A 2D numpy array with the unwrapped angles ensuring continuity.
    """

    x = three_dof_series[:, 1]
    xdot = np.diff(x)
    positive_indices = np.where(xdot > 0)[0]
    negative_indices = np.where(xdot < 0)[0]

    if len(negative_indices) == 0:
        return three_dof_series

    # first_idx, last_idx = negative_indices[0], negative_indices[-1]
    is_positive_segment_first = positive_indices[0] < negative_indices[0]
    is_negative_segment_first = positive_indices[0] > negative_indices[0]
    # assuming only one sign change
    if is_negative_segment_first:
        first_idx, last_idx = negative_indices[0], positive_indices[0] - 1
    if is_positive_segment_first:
        first_idx = positive_indices[-1] + 1
        end_first_negative_segment = np.where(np.diff(negative_indices) != 1)[0]
        last_idx = negative_indices[-1] if end_first_negative_segment.shape[0] == 0 else end_first_negative_segment[-1]

    unwrapped_series = three_dof_series.copy()
    for col in [0, 2]:
        unwrapped_series[:, col] = unwrap_segment(three_dof_series[:, col], first_idx + 1, last_idx + 1)

    # assuming only one sign change
    unwrapped_series[:, 1] = sign_change_segment(three_dof_series[:, 1], first_idx + 1, last_idx + 1)

    return unwrapped_series


def unwrap_segment(column: np.ndarray, first_idx: int, last_idx: int) -> np.ndarray:
    """
    Unwrap a segment of the angle series based on indices where the derivative is negative.
    """
    flipped_cropped = np.flip(column[: last_idx + 1])
    unwrapped_flipped = np.unwrap(flipped_cropped, period=180)
    unflipped = np.flip(unwrapped_flipped)
    unwrapped_end = np.unwrap(column[first_idx:], period=180)
    return np.concatenate((unflipped, unwrapped_end[last_idx - first_idx + 1 :]))


def sign_change_segment(column: np.ndarray, first_idx: int, last_idx: int) -> np.ndarray:
    """
    Adjust the signs in a segment of the angle series to ensure continuity.

    Parameters:
    column (np.ndarray): A 1D numpy array representing a single angle component.
    first_idx (int): The starting index of the segment to adjust.
    last_idx (int): The ending index of the segment to adjust.

    Returns:
    np.ndarray: The sign-adjusted segment of the angle series.
    """
    flipped_cropped = np.flip(column[: last_idx + 1])
    constant_sign_flipped_cropped = sign_change_array(flipped_cropped)
    unflipped = np.flip(constant_sign_flipped_cropped)
    unsigned_end = sign_change_array(column[first_idx:])
    return np.concatenate((unflipped, unsigned_end[last_idx - first_idx + 1 :]))


def sign_change_array(array: np.ndarray) -> np.ndarray:
    """
    Change the signs in an array to ensure all elements have the same sign as the first element.

    Parameters:
    array (np.ndarray): A 1D numpy array.

    Returns:
    np.ndarray: The sign-adjusted array.
    """
    first_sign = np.sign(array[0])
    array[np.sign(array) != first_sign] *= -1
    return array


def unwrap_segment_rotation_matrix(three_column: np.ndarray, first_idx: int, last_idx: int, seq: str) -> np.ndarray:
    """

    Parameters:
    column (np.ndarray): A 3D numpy array representing a three angle component.
    first_idx (int): The starting index of the segment to adjust.
    last_idx (int): The ending index of the segment to adjust.

    """
    flipped_cropped = np.flip(three_column[: last_idx + 1, :])
    unwrapped_flipped = unwrap_angles_from_euler_angles(angles_series=flipped_cropped, seq=seq)
    unflipped = np.flip(unwrapped_flipped)
    unwrapped_end = unwrap_angles_from_euler_angles(three_column[first_idx:, :], seq=seq)
    return np.concatenate((unflipped, unwrapped_end[last_idx - first_idx + 1 :]), axis=0)
