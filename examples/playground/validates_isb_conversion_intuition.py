"""
This example shows how to convert a rotation matrix from one coordinate system to another.
The rotation matrix is computed from the axes of the ISB coordinate system.
The rotation matrix is then converted to Euler angles, and the axes are permuted to match the sequence of the ISB axes.
Then we try to retrieve the euler angles from the ISB axes, and we compare the results.

"""

import numpy as np

# We can either do it with scipy or with biorbd
# Let's choose scipy as this is a more common toolbox
from scipy.spatial.transform import Rotation

from spartacus import compute_rotation_matrix_from_axes, flip_rotations

# simplification for matlabists
X = np.array([1, 0, 0])[:, np.newaxis]  # (3 x 1)
Y = np.array([0, 1, 0])[:, np.newaxis]  # (3 x 1)
Z = np.array([0, 0, 1])[:, np.newaxis]  # (3 x 1)

R_isb_isb = compute_rotation_matrix_from_axes(
    anterior_posterior_axis=X,
    infero_superior_axis=Y,
    medio_lateral_axis=Z,
)

print("R_isb_isb: ", R_isb_isb)
print("It gives the identity matrix as expected, as no rotation is performed as the base is already ISB")

# Let's consider another local coordinate system, with the following axes
R_isb_local = compute_rotation_matrix_from_axes(
    anterior_posterior_axis=Y,
    infero_superior_axis=Z,
    medio_lateral_axis=X,
)
R_local_isb = R_isb_local.T

print("R_isb_local: ", R_isb_local)
print("We get some permutation of the identity matrix")

# The sequence of the data
bad_sequence = "zyx"

# classic for sternoclavicular, acromioclavicular, and scapulothoracic joint
isb_sequence = "yxz"

# the sequence that matches previous identify euler angles (not expected based on intuition)
matching_sequence = "xzy"

print("R_isb_local: ", R_isb_local)
print("R_local_isb: ", R_local_isb)

# Let's build two rotation matrices R01 and R02, such that :
# a_in_0 = R01 @ a_in_1 and,
# b_in_0 = R02 @ b_in_2
# to emulate two segments with different orientations
# NOTE: intrinsic rotation are in uppercase in scipy
R01 = Rotation.from_euler("ZXY", np.array([0.1, 0.2, 0.3])).as_matrix()
R02 = Rotation.from_euler("ZXY", np.array([-0.09, -0.21, -0.301])).as_matrix()

# compute the rotation matrix between the two
R12 = R01.transpose() @ R02
print(R12)

# compute the euler angles of the rotation matrix with the sequence zxy
euler_scipy = Rotation.from_matrix(R12).as_euler(bad_sequence.upper())
print("Scipy euler angles of R12: ")
print(euler_scipy)


# applied the rotation matrix R_isb_local to R1 and R2
# R01_rotated = R01 @ R_isb_local.transpose()    # R01 @ R1_1'   ' stands for isb
# R02_rotated = R02 @ R_isb_local.transpose()    # R02 @ R2_2'   ' stands for isb

new_R = R_isb_local @ R12 @ R_isb_local.transpose()
#  new_R = R1'2'  ' stands for isb coordinate systems
# R_isb_local = R1'1   ' stands for isb
# R_isb_local = R2'2   ' stands for isb

# compute the euler angles of the rotated matrices
new_euler_scipy = Rotation.from_matrix(new_R).as_euler(isb_sequence.upper())
print("Scipy euler angles of new_R rotated:")
print(new_euler_scipy)

# print the two rotation matrices next to each other
print("previous rotation matrix | new rotation matrix")
print(np.hstack((R12, new_R)))

# check before if ratios are not too far from 1
if np.all(np.abs(new_euler_scipy - euler_scipy) < 0.0001):
    print("ratios are ok")
else:
    from spartacus.src.utils import flip_rotations

    print("ratios are not ok, trying to flip the rotations")
    new_euler_flipped = flip_rotations(new_euler_scipy, isb_sequence)
    if np.all(np.abs(new_euler_flipped - euler_scipy) < 0.0001):
        print("ratios are ok after flipping")
    else:
        raise RuntimeError(f"ratios are too far from 1: {new_euler_scipy / euler_scipy}")

# find the signs to apply to the euler angles to get the same result as the previous computation
print()
print("signs: ", new_euler_scipy / euler_scipy)
signs = np.sign(new_euler_scipy / euler_scipy)

# extra check try to rebuild the rotation matrix from the initial euler angles and the sign factors
extra_R_from_initial_euler_and_factors = Rotation.from_euler(isb_sequence.upper(), euler_scipy * signs).as_matrix()

print("extra_R_from_initial_euler_and_factors: ")
print(extra_R_from_initial_euler_and_factors)
print("new_R: ")
print(new_R)


# supplementary check
R_init_bad = R_isb_local.transpose()
R_init_isb = R_isb_isb

Rqq_bad = Rotation.from_euler("xzy", np.array([0.1, 0.2, 0.3])).as_matrix()
Rqq_isb = Rotation.from_euler("xzy", np.array([0.1, 0.2, 0.3])).as_matrix()

rotated_Rqq_bad = R_init_bad @ Rqq_bad  # R01 R_0_bad
rotated_Rqq_isb = R_init_isb @ Rqq_isb  # R02 R_0_isb
