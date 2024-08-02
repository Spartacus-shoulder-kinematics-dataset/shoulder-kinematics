import numpy as np


def unwrap_for_yxy_glenohumeral_joint(three_dof_series: np.ndarray) -> np.ndarray:
    """
    Unwrap the angles in a series of 3-DOF angles to ensure continuity,
    particularly handling the glenohumeral joint's potential rotations for glenohumeral yxy sequence

    Parameters:
    three_dof_series (np.ndarray): A 2D numpy array with each row representing a 3-DOF angle (y, x, y).

    Returns:
    np.ndarray: A 2D numpy array with the unwrapped angles ensuring continuity.
    """

    def unwrap_segment(column: np.ndarray, first_idx: int, last_idx: int) -> np.ndarray:
        """
        Unwrap a segment of the angle series based on indices where the derivative is negative.
        """
        flipped_cropped = np.flip(column[: last_idx + 1])
        unwrapped_flipped = np.unwrap(flipped_cropped, period=180)
        unflipped = np.flip(unwrapped_flipped)
        unwrapped_end = np.unwrap(column[first_idx:], period=180)
        return np.concatenate((unflipped, unwrapped_end[last_idx - first_idx + 1 :]))

    x = three_dof_series[:, 1]
    xdot = np.diff(x)
    positive_indices = np.where(xdot > 0)[0]
    negative_indices = np.where(xdot < 0)[0]

    if len(negative_indices) == 0:
        return three_dof_series

    first_idx, last_idx = negative_indices[0], negative_indices[-1]

    unwrapped_series = three_dof_series.copy()
    for col in [0, 2]:
        unwrapped_series[:, col] = unwrap_segment(three_dof_series[:, col], first_idx, last_idx)

    # If sign changing for xdot and x keep the sign of x on the negative_indices
    unwrapped_series[positive_indices, 1] *= -1

    return unwrapped_series


# Example usage
three_dof_series = np.array([[0, 0, 0], [1, 2, 1], [2, 3, 2], [3, 5, 3], [4, 1, 4]])
unwrapped_series = unwrap_for_yxy_glenohumeral_joint(three_dof_series)
print(unwrapped_series)
