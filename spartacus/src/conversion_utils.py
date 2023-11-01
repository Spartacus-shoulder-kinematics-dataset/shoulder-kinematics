from .biomech_system import BiomechCoordinateSystem
from .joint import Joint
from .enums import CartesianAxis, EulerSequence, JointType, Correction
from .kolz_matrices import get_kolz_rotation_matrix


def get_conversion_from_not_isb_to_isb_oriented(
    parent: BiomechCoordinateSystem,
    child: BiomechCoordinateSystem,
    joint: Joint,
) -> tuple[bool, tuple[int, int, int]]:
    """
    Check if the combination of the coordinates and the euler sequence is valid to be used, according to the ISB

    Parameters
    ----------
    parent_segment : BiomechCoordinateSystem
        The parent segment
    child_segment : BiomechCoordinateSystem
        The child segment
    joint : Joint
        The joint

    Returns
    -------
    tuple[bool, tuple[int,int,int]]
        usable : bool
            True if the combination is valid, False otherwise
        tuple[int,int,int]
            Sign to apply to the dataset to make it compatible with the ISB

    """

    # create an empty list of 7 element
    condition = [None] * 7
    # all the joints have the same rotation sequence for the ISB YXZ
    if joint.joint_type in (JointType.STERNO_CLAVICULAR, JointType.ACROMIO_CLAVICULAR, JointType.SCAPULO_THORACIC):
        # rotation -90° along X for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusZ
        condition[2] = parent.medio_lateral_axis == CartesianAxis.minusY
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusX
        condition[4] = child.infero_superior_axis == CartesianAxis.plusZ
        condition[5] = child.medio_lateral_axis == CartesianAxis.minusY
        condition[6] = joint.euler_sequence == EulerSequence.ZXY

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=z, x=x, z=-y")
            return True, (-1, 1, 1)

        # rotation 180° along X for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.minusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.minusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusX
        condition[4] = child.infero_superior_axis == CartesianAxis.minusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.minusZ
        condition[6] = joint.euler_sequence == EulerSequence.YXZ

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=-y, x=x, z=-z")
            return True, (-1, 1, -1)

        # rotation -270° along X for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.minusZ
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusY
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusX
        condition[4] = child.infero_superior_axis == CartesianAxis.minusZ
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusY
        condition[6] = joint.euler_sequence == EulerSequence.ZXY

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=-z, x=x, z=y")
            return True, (1, 1, -1)

        # Rotation -90° along Y for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusZ
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusX
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusZ
        condition[4] = child.infero_superior_axis == CartesianAxis.plusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusX
        condition[6] = joint.euler_sequence == EulerSequence.YZX

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=y, x=-z, z=x")
            return True, (1, -1, 1)

        # Rotation 180° along Y for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.minusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusX
        condition[4] = child.infero_superior_axis == CartesianAxis.plusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.minusZ
        condition[6] = joint.euler_sequence == EulerSequence.YXZ

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=y, x=-x, z=-z")
            return True, (1, -1, -1)

        # Rotation -270° along Y for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusZ
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.minusX
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusZ
        condition[4] = child.infero_superior_axis == CartesianAxis.plusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.minusX
        condition[6] = joint.euler_sequence == EulerSequence.YZX

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=y, x=z, z=-x")
            return True, (1, 1, -1)

        # Rotation 90° along Z for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusY
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusX
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusY
        condition[4] = child.infero_superior_axis == CartesianAxis.plusX
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusZ
        condition[6] = joint.euler_sequence == EulerSequence.XYZ

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=x, x=-y, z=z")
            return True, (1, -1, 1)

        # Rotation -90° along Z for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusY
        condition[1] = parent.infero_superior_axis == CartesianAxis.minusX
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusY
        condition[4] = child.infero_superior_axis == CartesianAxis.minusX
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusZ
        condition[6] = joint.euler_sequence == EulerSequence.XYZ

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=-x, x=y, z=z")
            return True, (-1, 1, 1)

        # Rotation 180° along Z for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.minusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusX
        condition[4] = child.infero_superior_axis == CartesianAxis.minusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusZ
        condition[6] = joint.euler_sequence == EulerSequence.XYZ

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=-y, x=-x, z=z")
            return True, (-1, -1, 1)

        # combined rotations +180 along z and +90 along x
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusZ
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusY
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusX
        condition[4] = child.infero_superior_axis == CartesianAxis.plusZ
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusY
        condition[6] = joint.euler_sequence == EulerSequence.ZXY

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXZ." "y=z, x=-x, z=y")
            return True, (1, -1, 1)

        # combined rotations +90 along x and +90 along y
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusY
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusZ
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusX
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusY
        condition[4] = child.infero_superior_axis == CartesianAxis.plusZ
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusX
        condition[6] = joint.euler_sequence == EulerSequence.ZYX

        if all(condition):
            # surprisingly this give the same angles, no sign change
            print("This is a valid combination, of the ISB sequence YXZ." "y=z, x=y, z=x")
            return True, (1, 1, 1)

        print("This is not a valid combination, of the ISB sequence YXZ.")
        return False, (0, 0, 0)

    # all the joints have the same rotation sequence for the ISB YXY
    elif joint.joint_type in (JointType.GLENO_HUMERAL, JointType.THORACO_HUMERAL):
        # Rotation -90° along X for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusZ
        condition[2] = parent.medio_lateral_axis == CartesianAxis.minusY
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusX
        condition[4] = child.infero_superior_axis == CartesianAxis.plusZ
        condition[5] = child.medio_lateral_axis == CartesianAxis.minusY
        condition[6] = joint.euler_sequence == EulerSequence.ZXZ

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=z, x=x, y=z")
            return True, (1, 1, 1)

        # Rotation 90° along X for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.minusZ
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusY
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusX
        condition[4] = child.infero_superior_axis == CartesianAxis.minusZ
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusY
        condition[6] = joint.euler_sequence == EulerSequence.ZXZ

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=-z, x=x, y=-z")
            return True, (-1, 1, -1)

        # Rotation 180° along X for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.minusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.minusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusX
        condition[4] = child.infero_superior_axis == CartesianAxis.minusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.minusZ
        condition[6] = joint.euler_sequence == EulerSequence.YXY

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=-y, x=x, y=-y")
            return True, (-1, 1, -1)

        # Rotation -90° along Y for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusZ
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusX
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusZ
        condition[4] = child.infero_superior_axis == CartesianAxis.plusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusX
        condition[6] = joint.euler_sequence == EulerSequence.YZY

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=y, x=-z, y=y")
            return True, (1, -1, 1)

        # Rotation 90° along Y for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusZ
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.minusX
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusZ
        condition[4] = child.infero_superior_axis == CartesianAxis.plusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.minusX
        condition[6] = joint.euler_sequence == EulerSequence.YZY

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=y, x=z, y=y")
            return True, (1, 1, 1)

        # Rotation 180° along Y for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.minusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusX
        condition[4] = child.infero_superior_axis == CartesianAxis.plusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.minusZ
        condition[6] = joint.euler_sequence == EulerSequence.YXY

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=y, x=-x, y=y")
            return True, (1, -1, 1)

        # Rotation -90° along Z for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.plusY
        condition[1] = parent.infero_superior_axis == CartesianAxis.minusX
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.plusY
        condition[4] = child.infero_superior_axis == CartesianAxis.minusX
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusZ
        condition[6] = joint.euler_sequence == EulerSequence.XYX

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=-x, x=y, y=-x")
            return True, (-1, 1, -1)

        # Rotation 90° along Z for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusY
        condition[1] = parent.infero_superior_axis == CartesianAxis.plusX
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusY
        condition[4] = child.infero_superior_axis == CartesianAxis.plusX
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusZ
        condition[6] = joint.euler_sequence == EulerSequence.XYX

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=x, x=-y, y=x")
            return True, (1, -1, 1)

        # Rotation 180° along Z for each segment coordinate system
        condition[0] = parent.anterior_posterior_axis == CartesianAxis.minusX
        condition[1] = parent.infero_superior_axis == CartesianAxis.minusY
        condition[2] = parent.medio_lateral_axis == CartesianAxis.plusZ
        condition[3] = child.anterior_posterior_axis == CartesianAxis.minusX
        condition[4] = child.infero_superior_axis == CartesianAxis.minusY
        condition[5] = child.medio_lateral_axis == CartesianAxis.plusZ
        condition[6] = joint.euler_sequence == EulerSequence.YXY

        if all(condition):
            print("This is a valid combination, of the ISB sequence YXY." " y=-y, x=-x, y=-y")
            return True, (-1, -1, -1)

        return False, (0, 0, 0)

    else:
        raise ValueError(
            "The JointType is not supported. Please use:"
            "JointType.GLENO_HUMERAL, JointType.ACROMIO_CLAVICULAR,"
            "JointType.STERNO_CLAVICULAR, JointType.THORACO_HUMERAL,"
            "or JointType.SCAPULO_THORACIC"
        )


def convert_rotation_matrix_from_one_coordinate_system_to_another(
    bsys: BiomechCoordinateSystem,
    initial_sequence: EulerSequence,
    sequence_wanted: EulerSequence,
    child_extra_correction: Correction = None,
    parent_extra_correction: Correction = None,
) -> tuple[bool, tuple[int, int, int]]:
    """
    This function converts the current euler angles into a desired euler sequence.

    Parameters
    ----------
    bsys: BiomechCoordinateSystem
        The biomechanical coordinate system of the segment
    initial_sequence: EulerSequence
        The euler sequence of the rotation matrix
    sequence_wanted: EulerSequence
        The euler sequence of the rotation matrix wanted, e.g. ISB sequence
    child_extra_correction: Correction
        The correction to apply to the child segment
    parent_extra_correction: Correction
        The correction to apply to the parent segment

    Returns
    -------
    bool
        Whether the conversion is possible with sign factors
    tuple[int, int, int]
        The sign factors to apply to the euler angles to get the desired euler sequence
    """
    initial_sequence = initial_sequence.value
    sequence_wanted = sequence_wanted.value  # most of the time, it will be the ISB sequence

    R_isb_local = mat_2_rotation(bsys.get_rotation_matrix()).to_array()

    # Let's build two rotation matrices R01 and R02, such that a_in_0 = R01 @ a_in_1 and b_in_0 = R02 @ b_in_2
    # to emulate two segments with different orientations
    R01 = biorbd.Rotation.fromEulerAngles(rot=np.array([0.5, -0.8, 1.2]), seq="zxy").to_array()  # random values
    R02 = biorbd.Rotation.fromEulerAngles(rot=np.array([-0.01, 0.02, -0.03]), seq="zxy").to_array()  # ra

    # compute the rotation matrix between the two
    R12 = R01.transpose() @ R02
    # print(R12)

    # compute the euler angles of the rotation matrix with the sequence zxy
    euler = biorbd.Rotation.toEulerAngles(mat_2_rotation(R12), initial_sequence).to_array()
    # print("euler angles")
    # print(euler)
    # if we want to change if for scipy
    # euler_scipy = Rotation.from_matrix(R12).as_euler(bad_sequence.upper())

    # applied the rotation matrix R to R1 and R2
    #  ---  Deprecated --- not false but less generic
    # R01_rotated = R01 @ R_isb_local.transpose()
    # R02_rotated = R02 @ R_isb_local.transpose()
    # new_R = R01_rotated.transpose() @ R02_rotated
    #  ---  New way --- more generic
    # 1 : parent
    # 2 : child
    new_R = R_isb_local @ R12 @ R_isb_local.transpose()

    # extra corrections - Kolz
    if child_extra_correction is not None:
        print(f"I applied a correction of {child_extra_correction} to the child segment")
        R_child_correction = get_kolz_rotation_matrix(child_extra_correction, orthonormalize=True).T
        new_R = new_R @ R_child_correction

    if parent_extra_correction is not None:
        print(f"I applied a correction of {parent_extra_correction} to the parent segment")
        R_parent_correction = get_kolz_rotation_matrix(parent_extra_correction, orthonormalize=True)
        new_R = R_parent_correction @ new_R

    # compute the euler angles of the rotated matrices
    new_euler = biorbd.Rotation.toEulerAngles(mat_2_rotation(new_R), sequence_wanted).to_array()
    # print("euler angles of new_R rotated")
    # print(euler1)
    # if we want to change if for scipy
    # new_euler_scipy = Rotation.from_matrix(new_R).as_euler(isb_sequence.upper())

    # check before if ratios are not too far from 1
    ratio = new_euler / euler
    # check if the ratios with flipped euler angles are not too far from 1
    new_euler_flipped = flip_rotations(new_euler, sequence_wanted)
    ratio_flipped = new_euler_flipped / euler
    if not np.any(np.abs(ratio) < 0.999) and not np.any(np.abs(ratio) > 1.001):
        print("ratios are ok")
    elif not np.any(np.abs(ratio_flipped) < 0.999) and not np.any(np.abs(ratio_flipped) > 1.001):
        print("ratios are ok with flipped euler angles")
        new_euler = new_euler_flipped
    else:
        # raise RuntimeError(f"ratios are too far from 1: {ratio}")
        return False, (0, 0, 0)

    # find the signs to apply to the euler angles to get the same result as the previous computation
    signs = np.sign(ratio)

    # extra check try to rebuild the rotation matrix from the initial euler angles and the sign factors
    extra_R_from_initial_euler_and_factors = biorbd.Rotation.fromEulerAngles(
        rot=euler * signs, seq=sequence_wanted
    ).to_array()
    if not np.allclose(new_R, extra_R_from_initial_euler_and_factors):
        raise RuntimeError("The rebuilt rotation matrix is not the same as the original one")

    # print("conversion factors to apply to the euler angles are:")
    # print(signs)

    return True, tuple(signs)


def get_conversion_from_not_isb_to_isb_oriented_v2(
    parent: BiomechCoordinateSystem,
    child: BiomechCoordinateSystem,
    joint: Joint,
) -> tuple[bool, callable]:
    """
    Get the conversion factor to convert the rotation matrix from the parent segment
    to the child segment to the ISB sequence

    Parameters
    ----------
    parent : BiomechCoordinateSystem
        The parent segment coordinate system
    child : BiomechCoordinateSystem
        The child segment coordinate system
    joint : Joint
        The joint type

    Returns
    -------
    tuple(bool, callable)
        bool
            True if the biomechanical coordinate system is compatible with ISB
        tuple
            The conversion factor for dof1, dof2, dof3 of euler angles

    """
    if joint.joint_type in (JointType.STERNO_CLAVICULAR, JointType.ACROMIO_CLAVICULAR, JointType.SCAPULO_THORACIC):
        sequence_wanted = EulerSequence.YXZ
        # check that we have three different letters in the sequence
        if len(set(joint.euler_sequence.value)) != 3:
            raise RuntimeError(
                "The euler sequence of the joint must have three different letters to be able to convert with factors 1"
                f"or -1 to the ISB sequence {sequence_wanted.value}, but the sequence of the joint is"
                f" {joint.euler_sequence.value}"
            )
    elif joint.joint_type in (JointType.GLENO_HUMERAL, JointType.THORACO_HUMERAL):
        sequence_wanted = EulerSequence.YXY
        # check that the sequence in joint.euler_sequence as the same two letters for the first and third rotations
        if joint.euler_sequence.value[0] != joint.euler_sequence.value[2]:
            raise RuntimeError(
                "The euler sequence of the joint must have the same two letters for the first and third rotations"
                f"to be able to convert with factors 1 or -1 to the ISB sequence {sequence_wanted.value},"
                f" but the sequence of the joint is {joint.euler_sequence.value}"
            )
    else:
        raise RuntimeError(
            "The joint type must be JointType.STERNO_CLAVICULAR, JointType.ACROMIO_CLAVICULAR,"
            "JointType.SCAPULO_THORACIC, JointType.GLENO_HUMERAL, JointType.THORACO_HUMERAL"
        )

    the_tuple = convert_rotation_matrix_from_one_coordinate_system_to_another(
        parent,  # sending only the parent segment since the two segments have the same orientation
        joint.euler_sequence,
        sequence_wanted,
    )

    return True, the_tuple
