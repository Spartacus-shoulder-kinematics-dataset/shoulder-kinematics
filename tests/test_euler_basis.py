import numpy as np
from spartacus import (
    rotation_x,
    rotation_y,
    rotation_z,
    euler_axes_from_rotation_matrices,
    EulerSequence,
    from_jcs_to_parent_frame,
)


def test_euler_basis():
    R_0_parent = np.eye(3)
    R_0_child = rotation_x(0.2) @ rotation_y(0.5) @ rotation_z(1)
    euler_axes = euler_axes_from_rotation_matrices(
        R_0_parent=R_0_parent,
        R_0_child=R_0_child,
        sequence=EulerSequence.XYZ,
        axes_source_frame="mixed",
    )
    R_proximal_euler = np.vstack(euler_axes)

    expected_euler_axes = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.98006658, 0.19866933], [0.47942554, -0.17434874, 0.86008934]]
    )
    expected_arcsin_1 = 0.5000000015905022
    expected_arcsin_2 = 0.19999999918876815

    np.testing.assert_almost_equal(R_proximal_euler, expected_euler_axes, decimal=8)
    np.testing.assert_almost_equal(np.arcsin(R_proximal_euler[2, 0]), expected_arcsin_1, decimal=8)
    np.testing.assert_almost_equal(np.arcsin(R_proximal_euler[1, -1]), expected_arcsin_2, decimal=8)


def test_euler_basis_translation_correction():
    t = np.array([1, 2, 3])
    rot = np.array([0.2, 0.5, 1])
    seq = EulerSequence.XYZ

    tt = from_jcs_to_parent_frame(t, rot, seq)

    expected_tt = np.array([1.        , 1.38951922, 3.21226559])
    np.testing.assert_almost_equal(tt, expected_tt, decimal=8)
