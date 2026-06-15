import numpy as np

# We can either do it with scipy or with biorbd
# Let's choose scipy as this is a more common toolbox
from scipy.spatial.transform import Rotation

from spartacus import compute_rotation_matrix_from_axes

X = np.array([1, 0, 0])[:, np.newaxis]  # (3 x 1)
Y = np.array([0, 1, 0])[:, np.newaxis]  # (3 x 1)
Z = np.array([0, 0, 1])[:, np.newaxis]  # (3 x 1)


def test_rotation_matrix_from_axes_isb():
    # simplification for matlabists

    R_isb_isb = compute_rotation_matrix_from_axes(
        anterior_posterior_axis=X,
        infero_superior_axis=Y,
        medio_lateral_axis=Z,
    )

    assert np.allclose(R_isb_isb, np.eye(3))


def test_rotation_matrix_from_axes_not_isb_yzx():
    R_isb_local = compute_rotation_matrix_from_axes(
        anterior_posterior_axis=Y,
        infero_superior_axis=Z,
        medio_lateral_axis=X,
    )

    assert np.allclose(R_isb_local, np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))

    # Does it mean what it should mean?
    # Let's consider a vector in the original local coordinate system
    v_in_local = np.array([1, 2, 3])[:, np.newaxis]
    v_in_isb_expected = np.array([2, 3, 1])[:, np.newaxis]

    # Let's rotate it to the ISB coordinate system
    v_in_isb = R_isb_local @ v_in_local

    assert np.allclose(v_in_isb, v_in_isb_expected)


def test_rotation_matrix_from_axes_not_isb_zyx():
    R_isb_local = compute_rotation_matrix_from_axes(
        anterior_posterior_axis=Z,
        infero_superior_axis=Y,
        medio_lateral_axis=X,
    )

    assert np.allclose(R_isb_local, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))

    # Does it mean what it should mean?
    # Let's consider a vector in the original local coordinate system
    v_in_local = np.array([1, 2, 3])[:, np.newaxis]
    v_in_isb_expected = np.array([3, 2, 1])[:, np.newaxis]

    # Let's rotate it to the ISB coordinate system
    v_in_isb = R_isb_local @ v_in_local

    assert np.allclose(v_in_isb, v_in_isb_expected)


def test_rotation_matrix_from_axes_not_isb_yxz():
    R_isb_local = compute_rotation_matrix_from_axes(
        anterior_posterior_axis=Y,
        infero_superior_axis=-X,
        medio_lateral_axis=Z,
    )

    assert np.allclose(R_isb_local, np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

    # Does it mean what it should mean?
    # Let's consider a vector in the original local coordinate system
    v_in_local = np.array([1, 2, 3])[:, np.newaxis]
    v_in_isb_expected = np.array([2, -1, 3])[:, np.newaxis]

    # Let's rotate it to the ISB coordinate system
    v_in_isb = R_isb_local @ v_in_local

    assert np.allclose(v_in_isb, v_in_isb_expected)


def test_from_two_segment_local_yzx_joint_zyx_to_yxz():
    # Let's consider another local coordinate system, with the following axes
    R_isb_local = compute_rotation_matrix_from_axes(
        anterior_posterior_axis=Y,
        infero_superior_axis=Z,
        medio_lateral_axis=X,
    )
    R_local_isb = R_isb_local.T
    print("R_isb_local: ", R_isb_local)
    print("R_local_isb: ", R_local_isb)

    # The sequence of the data, with which the data would have been communicated
    bad_sequence = "zyx"

    # classic for sternoclavicular, acromioclavicular, and scapulothoracic joint
    isb_sequence = "yxz"

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

    # Fastest way to do it
    new_R = R_isb_local @ R12 @ R_isb_local.transpose()

    print("new_R: ", new_R)

    assert np.allclose(
        new_R,
        np.array(
            [
                [0.89987124, 0.43205263, -0.05968471],
                [-0.32288006, 0.75189386, 0.57480788],
                [0.29322382, -0.49798208, 0.81610884],
            ]
        ),
    )

    #  new_R = R1'2'  ' stands for isb coordinate systems
    # R_isb_local = R1'1   ' stands for isb
    # R_isb_local = R2'2   ' stands for isb

    # compute the euler angles of the rotated matrices
    new_euler_scipy = Rotation.from_matrix(new_R).as_euler(isb_sequence.upper())
    print("Scipy euler angles of new_R rotated:")
    print(new_euler_scipy)
    # In this case, we expect the same euler angles as the original rotation matrix
    # because we rotated both segments with the same rotation matrix without changing
    # the Euler sequence

    assert np.allclose(new_euler_scipy, euler_scipy)

    # find the signs to apply to the euler angles to get the same result as the previous computation
    print("signs: ", new_euler_scipy / euler_scipy)
    signs = np.sign(new_euler_scipy / euler_scipy)
    # extra check try to rebuild the rotation matrix from the initial euler angles and the sign factors
    extra_R_from_initial_euler_and_factors = Rotation.from_euler(isb_sequence.upper(), euler_scipy * signs).as_matrix()

    assert np.allclose(extra_R_from_initial_euler_and_factors, new_R)


def test_from_two_segment_local_xyz_joint_zxy_to_yxz():
    # Let's consider another local coordinate system, with the following axes
    R_isb_local = compute_rotation_matrix_from_axes(
        anterior_posterior_axis=X,
        infero_superior_axis=Z,
        medio_lateral_axis=-Y,
    )
    R_local_isb = R_isb_local.T
    print("R_isb_local: ", R_isb_local)
    print("R_local_isb: ", R_local_isb)

    # The sequence of the data, with which the data would have been communicated
    bad_sequence = "zxy"

    # wanted just for testing if I understand my test
    isb_sequence = "yxz"
    expected_signs = np.array([1, 1, -1])  # idk yet

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

    # Fastest way to do it
    new_R = R_isb_local @ R12 @ R_isb_local.transpose()

    #  new_R = R1'2'  ' stands for isb coordinate systems
    # R_isb_local = R1'1   ' stands for isb
    # R_isb_local = R2'2   ' stands for isb

    # compute the euler angles of the rotated matrices
    new_euler_scipy = Rotation.from_matrix(new_R).as_euler(isb_sequence.upper())
    print("Scipy euler angles of new_R rotated:")
    print(new_euler_scipy)
    # In this case, we expect the same euler angles as the original rotation matrix
    # because we rotated both segments with the same rotation matrix without changing
    # the Euler sequence

    assert np.allclose(new_euler_scipy, euler_scipy * expected_signs)

    # find the signs to apply to the euler angles to get the same result as the previous computation
    print("signs: ", new_euler_scipy / euler_scipy)
    signs = np.sign(new_euler_scipy / euler_scipy)
    # extra check try to rebuild the rotation matrix from the initial euler angles and the sign factors
    extra_R_from_initial_euler_and_factors = Rotation.from_euler(isb_sequence.upper(), euler_scipy * signs).as_matrix()

    assert np.allclose(extra_R_from_initial_euler_and_factors, new_R)


def test_from_two_segment_local_yxz_joint_xzy_to_yxz():
    R_isb_local = compute_rotation_matrix_from_axes(
        anterior_posterior_axis=X,
        infero_superior_axis=Z,
        medio_lateral_axis=-Y,
    )
    R_local_isb = R_isb_local.T
    print("R_isb_local: ", R_isb_local)
    print("R_local_isb: ", R_local_isb)

    # The sequence of the data, with which the data would have been communicated
    bad_sequence = "xzy"

    # wanted just for testing if I understand my test
    isb_sequence = "xyz"
    expected_signs = np.array([1, 1, -1])  # idk yet

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

    # Fastest way to do it
    new_R = R_isb_local @ R12 @ R_isb_local.transpose()

    #  new_R = R1'2'  ' stands for isb coordinate systems
    # R_isb_local = R1'1   ' stands for isb
    # R_isb_local = R2'2   ' stands for isb

    # compute the euler angles of the rotated matrices
    new_euler_scipy = Rotation.from_matrix(new_R).as_euler(isb_sequence.upper())
    print("Scipy euler angles of new_R rotated:")
    print(new_euler_scipy)
    # In this case, we expect the same euler angles as the original rotation matrix
    # because we rotated both segments with the same rotation matrix without changing
    # the Euler sequence

    assert np.allclose(new_euler_scipy, euler_scipy * expected_signs)

    # find the signs to apply to the euler angles to get the same result as the previous computation
    print("signs: ", new_euler_scipy / euler_scipy)
    signs = np.sign(new_euler_scipy / euler_scipy)
    # extra check try to rebuild the rotation matrix from the initial euler angles and the sign factors
    extra_R_from_initial_euler_and_factors = Rotation.from_euler(isb_sequence.upper(), euler_scipy * signs).as_matrix()

    assert np.allclose(extra_R_from_initial_euler_and_factors, new_R)
