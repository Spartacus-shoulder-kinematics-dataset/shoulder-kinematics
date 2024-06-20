"""
The isb euler sequence is supposed to be equivalent when parent and child frames are rotated from the isb norm, but lead to the same angles.
Examples are tested below for the sterno-clavicular, scapulo-thoracic and gleno-humeral joints.
"""

import numpy as np

from spartacus import EulerSequence, Segment, JointType, BiomechCoordinateSystem, BiomechDirection, Joint
from spartacus.src.corrections.angle_conversion_callbacks import convert_euler_angles_and_frames_to_isb

rot1 = 0.2
rot2 = 0.3
rot3 = 0.4


def test_isb_sterno_clav():

    isb_seq = EulerSequence.isb_from_joint_type(JointType.STERNO_CLAVICULAR).value
    seq = EulerSequence.YXZ

    thorax_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.THORAX,
        x=BiomechDirection.PlusPosteroAnterior,
        y=BiomechDirection.PlusInferoSuperior,
        z=BiomechDirection.PlusMedioLateral,
    )

    clav_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.CLAVICLE,
        x=BiomechDirection.PlusPosteroAnterior,
        y=BiomechDirection.PlusInferoSuperior,
        z=BiomechDirection.PlusMedioLateral,
    )

    new_angles = convert_euler_angles_and_frames_to_isb(
        previous_sequence_str=seq.value,
        new_sequence_str=isb_seq,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        bsys_parent=thorax_sys,
        bsys_child=clav_sys,
    )

    is_first_angle_equal = np.round(new_angles[0], 5) == rot1
    is_second_angle_equal = np.round(new_angles[1], 5) == rot2
    is_third_angle_equal = np.round(new_angles[2], 5) == rot3

    assert is_first_angle_equal and is_second_angle_equal and is_third_angle_equal

    sterno_clav_joint = Joint(
        joint_type=JointType.STERNO_CLAVICULAR,
        euler_sequence=seq,
        translation_origin=None,
        translation_frame=None,
        parent_segment=thorax_sys,
        child_segment=clav_sys,
    )
    assert sterno_clav_joint.is_euler_sequence_equivalent_to_isb == True

    # rotated framed and switched euler sequence
    seq = EulerSequence.ZYX
    thorax_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.THORAX,
        x=BiomechDirection.PlusMedioLateral,
        y=BiomechDirection.PlusPosteroAnterior,
        z=BiomechDirection.PlusInferoSuperior,
    )

    clav_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.CLAVICLE,
        x=BiomechDirection.PlusMedioLateral,
        y=BiomechDirection.PlusPosteroAnterior,
        z=BiomechDirection.PlusInferoSuperior,
    )

    new_angles = convert_euler_angles_and_frames_to_isb(
        previous_sequence_str=seq.value,
        new_sequence_str=isb_seq,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        bsys_parent=thorax_sys,
        bsys_child=clav_sys,
    )

    is_first_angle_equal = np.round(new_angles[0], 5) == rot1
    is_second_angle_equal = np.round(new_angles[1], 5) == rot2
    is_third_angle_equal = np.round(new_angles[2], 5) == rot3

    assert is_first_angle_equal and is_second_angle_equal and is_third_angle_equal

    sterno_clav_joint = Joint(
        joint_type=JointType.STERNO_CLAVICULAR,
        euler_sequence=seq,
        translation_origin=None,
        translation_frame=None,
        parent_segment=thorax_sys,
        child_segment=clav_sys,
    )
    assert sterno_clav_joint.is_euler_sequence_equivalent_to_isb == True


def test_isb_scapulothoracic():
    joint = JointType.SCAPULO_THORACIC
    isb_seq = EulerSequence.isb_from_joint_type(joint).value
    seq = EulerSequence.YXZ

    thorax_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.THORAX,
        x=BiomechDirection.PlusPosteroAnterior,
        y=BiomechDirection.PlusInferoSuperior,
        z=BiomechDirection.PlusMedioLateral,
    )

    scapula_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.SCAPULA,
        x=BiomechDirection.PlusPosteroAnterior,
        y=BiomechDirection.PlusInferoSuperior,
        z=BiomechDirection.PlusMedioLateral,
    )

    new_angles = convert_euler_angles_and_frames_to_isb(
        previous_sequence_str=seq.value,
        new_sequence_str=isb_seq,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        bsys_parent=thorax_sys,
        bsys_child=scapula_sys,
    )

    is_first_angle_equal = np.round(new_angles[0], 5) == rot1
    is_second_angle_equal = np.round(new_angles[1], 5) == rot2
    is_third_angle_equal = np.round(new_angles[2], 5) == rot3

    assert is_first_angle_equal and is_second_angle_equal and is_third_angle_equal

    sterno_clav_joint = Joint(
        joint_type=joint,
        euler_sequence=seq,
        translation_origin=None,
        translation_frame=None,
        parent_segment=thorax_sys,
        child_segment=scapula_sys,
    )
    assert sterno_clav_joint.is_euler_sequence_equivalent_to_isb == True

    # rotated framed and switched euler sequence
    seq = EulerSequence.ZYX
    thorax_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.THORAX,
        x=BiomechDirection.PlusMedioLateral,
        y=BiomechDirection.PlusPosteroAnterior,
        z=BiomechDirection.PlusInferoSuperior,
    )

    scapula_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.SCAPULA,
        x=BiomechDirection.PlusMedioLateral,
        y=BiomechDirection.PlusPosteroAnterior,
        z=BiomechDirection.PlusInferoSuperior,
    )

    new_angles = convert_euler_angles_and_frames_to_isb(
        previous_sequence_str=seq.value,
        new_sequence_str=isb_seq,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        bsys_parent=thorax_sys,
        bsys_child=scapula_sys,
    )

    is_first_angle_equal = np.round(new_angles[0], 5) == rot1
    is_second_angle_equal = np.round(new_angles[1], 5) == rot2
    is_third_angle_equal = np.round(new_angles[2], 5) == rot3

    assert is_first_angle_equal and is_second_angle_equal and is_third_angle_equal

    scapulothoracic = Joint(
        joint_type=JointType.STERNO_CLAVICULAR,
        euler_sequence=seq,
        translation_origin=None,
        translation_frame=None,
        parent_segment=thorax_sys,
        child_segment=scapula_sys,
    )
    assert scapulothoracic.is_euler_sequence_equivalent_to_isb == True

    scapulothoracic = Joint(
        joint_type=JointType.STERNO_CLAVICULAR,
        euler_sequence=EulerSequence.XYX,
        translation_origin=None,
        translation_frame=None,
        parent_segment=thorax_sys,
        child_segment=scapula_sys,
    )
    assert scapulothoracic.is_euler_sequence_equivalent_to_isb == False


def test_isb_gh():
    joint = JointType.GLENO_HUMERAL
    isb_seq = EulerSequence.isb_from_joint_type(joint).value
    seq = EulerSequence.YXY

    scapula_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.SCAPULA,
        x=BiomechDirection.PlusPosteroAnterior,
        y=BiomechDirection.PlusInferoSuperior,
        z=BiomechDirection.PlusMedioLateral,
    )

    humerus_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.HUMERUS,
        x=BiomechDirection.PlusPosteroAnterior,
        y=BiomechDirection.PlusInferoSuperior,
        z=BiomechDirection.PlusMedioLateral,
    )

    new_angles = convert_euler_angles_and_frames_to_isb(
        previous_sequence_str=seq.value,
        new_sequence_str=isb_seq,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        bsys_parent=scapula_sys,
        bsys_child=humerus_sys,
    )

    is_first_angle_equal = np.round(new_angles[0], 5) == rot1
    is_second_angle_equal = np.round(new_angles[1], 5) == rot2
    is_third_angle_equal = np.round(new_angles[2], 5) == rot3

    assert is_first_angle_equal and is_second_angle_equal and is_third_angle_equal

    gh = Joint(
        joint_type=joint,
        euler_sequence=seq,
        translation_origin=None,
        translation_frame=None,
        parent_segment=scapula_sys,
        child_segment=humerus_sys,
    )
    assert gh.is_euler_sequence_equivalent_to_isb == True

    # rotated framed and switched euler sequence
    seq = EulerSequence.ZYZ

    scapula_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.SCAPULA,
        x=BiomechDirection.PlusMedioLateral,
        y=BiomechDirection.PlusPosteroAnterior,
        z=BiomechDirection.PlusInferoSuperior,
    )

    humerus_sys = BiomechCoordinateSystem.from_biomech_directions(
        segment=Segment.HUMERUS,
        x=BiomechDirection.PlusMedioLateral,
        y=BiomechDirection.PlusPosteroAnterior,
        z=BiomechDirection.PlusInferoSuperior,
    )

    new_angles = convert_euler_angles_and_frames_to_isb(
        previous_sequence_str=seq.value,
        new_sequence_str=isb_seq,
        rot1=rot1,
        rot2=rot2,
        rot3=rot3,
        bsys_parent=scapula_sys,
        bsys_child=humerus_sys,
    )

    is_first_angle_equal = np.round(new_angles[0], 5) == rot1
    is_second_angle_equal = np.round(new_angles[1], 5) == rot2
    is_third_angle_equal = np.round(new_angles[2], 5) == rot3

    assert is_first_angle_equal and is_second_angle_equal and is_third_angle_equal

    gh = Joint(
        joint_type=JointType.GLENO_HUMERAL,
        euler_sequence=seq,
        translation_origin=None,
        translation_frame=None,
        parent_segment=scapula_sys,
        child_segment=humerus_sys,
    )
    assert gh.is_euler_sequence_equivalent_to_isb == True

    scapulothoracic = Joint(
        joint_type=JointType.STERNO_CLAVICULAR,
        euler_sequence=EulerSequence.XZY,
        translation_origin=None,
        translation_frame=None,
        parent_segment=scapula_sys,
        child_segment=humerus_sys,
    )
    assert scapulothoracic.is_euler_sequence_equivalent_to_isb == False
