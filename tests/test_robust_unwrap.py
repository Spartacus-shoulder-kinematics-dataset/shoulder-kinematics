import numpy as np
from spartacus.src.corrections.robust_unwrap import unwrap_rotation_matrix_from_euler_angles
import biorbd


def test_robust_unwrap():

    angles = np.array([-103.34, 21.58, 89.35]) / 180 * np.pi
    angles_init = np.array([75.74, -21.98, -88]) / 180 * np.pi

    print(angles_init * 180 / np.pi)

    new_angles = unwrap_rotation_matrix_from_euler_angles(angles=angles, seq="yxy", angles_init=angles_init)

    print("Expected Rotation Matrix: \n", biorbd.Rotation.fromEulerAngles(rot=angles, seq="yxy").to_array())
    print("New Rotation Matrix: \n", biorbd.Rotation.fromEulerAngles(rot=new_angles, seq="yxy").to_array())

    np.testing.assert_almost_equal(
        biorbd.Rotation.fromEulerAngles(rot=new_angles, seq="yxy").to_array(),
        biorbd.Rotation.fromEulerAngles(rot=angles, seq="yxy").to_array(),
    )


def test_robust_unwrap2():

    angles = np.array([-124.77, -43.616, -144.99]) / 180 * np.pi
    angles_init = np.array([0, -43.616, 1]) / 180 * np.pi

    print(angles_init * 180 / np.pi)

    new_angles = unwrap_rotation_matrix_from_euler_angles(angles=angles, seq="yxy", angles_init=angles_init)

    print(new_angles * 180 / np.pi)
    print("Expected Rotation Matrix: \n", biorbd.Rotation.fromEulerAngles(rot=angles, seq="yxy").to_array())
    print("New Rotation Matrix: \n", biorbd.Rotation.fromEulerAngles(rot=new_angles, seq="yxy").to_array())

    # np.testing.assert_almost_equal(
    #     biorbd.Rotation.fromEulerAngles(rot=new_angles, seq="yxy").to_array(),
    #     biorbd.Rotation.fromEulerAngles(rot=angles, seq="yxy").to_array(),
    # )
