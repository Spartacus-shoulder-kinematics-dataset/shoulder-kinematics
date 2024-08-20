import numpy as np
from spartacus import rotation_x, rotation_z, rotation_y, euler_axes_from_rotation_matrices, EulerSequence


def main():
    R_0_parent = np.eye(3)
    R_0_child = rotation_x(0.2) @ rotation_y(0.5) @ rotation_z(1)
    euler_axes = euler_axes_from_rotation_matrices(
        R_0_parent=R_0_parent,
        R_0_child=R_0_child,
        sequence=EulerSequence.XYZ,
        axes_source_frame="mixed",
    )
    print(np.vstack(euler_axes))
    print(np.arcsin(0.47942554))
    print(np.arcsin(0.19866933))


if __name__ == "__main__":
    main()
