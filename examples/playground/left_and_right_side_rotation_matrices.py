"""
This example shows how to convert a rotation matrix from a right-handed coordinate system to a left-handed coordinate system
or vice versa. We verify that the expected Euler angles signed are obtained after the conversion from one side to the other.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from spartacus import flip_rotations
from spartacus.src.corrections.angle_conversion_callbacks import to_left_handed_frame


def main():
    seq = ["ZXY", "YXZ", "XYZ", "ZYX", "XZY", "YZX", "YXY", "ZXZ", "XYX", "ZYZ"]

    for s in seq:
        print(f"Sequence: {s}")
        R01 = Rotation.from_euler(s, np.array([0.1, 0.2, 0.3])).as_matrix()
        signs = np.ones(3)
        for i, letter in enumerate(s):
            if letter in ("X", "Y"):
                signs[i] = -1
        print(signs)
        R02 = Rotation.from_euler(s, np.array([0.1, -0.2, -0.3])).as_matrix()

        R01_left = to_left_handed_frame(matrix=R01)

        euler_angles = Rotation.from_matrix(R01_left).as_euler(s)
        seq_has_two_same_letters = len(set(s)) != len(s)
        if seq_has_two_same_letters:
            euler_angles = flip_rotations(euler_angles, s)
            print(f"Flipped from {Rotation.from_matrix(R01_left).as_euler(s)} to {euler_angles}")

        np.testing.assert_allclose(euler_angles, [signs[0] * 0.1, signs[1] * 0.2, signs[2] * 0.3])
        print(
            f"It works for sequence {s}: {Rotation.from_matrix(R01_left).as_euler(s)} "
            f"and expected {[signs[0] * 0.1, signs[1] * 0.2, signs[2] * 0.3]}"
        )


if __name__ == "__main__":
    main()
