import pytest

from spartacus import (
    AnatomicalLandmark,
    Segment,
)


def test_checks():
    print("todo")
    # # A very standard example of the sterno-clavicular joint, already in ISB
    # print(" -- Sterno-clavicular joint -- ISB")
    # thorax = BiomechCoordinateSystem(
    #     antero_posterior_axis=CartesianAxis.plusX,
    #     infero_superior_axis=CartesianAxis.plusY,
    #     medio_lateral_axis=CartesianAxis.plusZ,
    #     segment=None,
    #     origin=None,
    # )
    # assert thorax.is_isb_oriented == True
    # clavicle = BiomechCoordinateSystem(
    #     antero_posterior_axis=CartesianAxis.plusX,
    #     infero_superior_axis=CartesianAxis.plusY,
    #     medio_lateral_axis=CartesianAxis.plusZ,
    #     segment=None,
    #     origin=None,
    # )
    # assert clavicle.is_isb_oriented == True
    # sterno_clav = Joint(
    #     joint_type=JointType.STERNO_CLAVICULAR,
    #     euler_sequence=EulerSequence.YXZ,
    #     translation_frame=None,
    #     translation_origin=None,
    # )
    # print(sterno_clav.is_joint_sequence_isb)
    # assert sterno_clav.is_joint_sequence_isb == True
    #
    # # A not standard example of the sterno-clavicular joint, not in ISB
    # print(" -- Sterno-clavicular joint -- not ISB")
    # thorax = BiomechCoordinateSystem(
    #     antero_posterior_axis=CartesianAxis.plusX,
    #     infero_superior_axis=CartesianAxis.plusY,
    #     medio_lateral_axis=CartesianAxis.plusZ,
    #     segment=None,
    #     origin=None,
    # )
    # print(thorax.is_isb_oriented)
    # assert thorax.is_isb_oriented == True
    # clavicle = BiomechCoordinateSystem(
    #     antero_posterior_axis=CartesianAxis.minusY,
    #     infero_superior_axis=CartesianAxis.plusX,
    #     medio_lateral_axis=CartesianAxis.plusZ,
    #     segment=None,
    #     origin=None,
    # )
    # print(clavicle.is_isb_oriented)
    # assert clavicle.is_isb_oriented == False
    # sterno_clav = Joint(
    #     joint_type=JointType.STERNO_CLAVICULAR,
    #     euler_sequence=EulerSequence.YXZ,
    #     translation_frame=None,
    #     translation_origin=None,
    # )
    # print(sterno_clav.is_joint_sequence_isb)
    # assert sterno_clav.is_joint_sequence_isb == True
    #
    # # A not standard example of the sterno-clavicular joint, not in ISB, but compatible with ISB
    # print(" -- Sterno-clavicular joint -- not ISB but compatible")
    # thorax = BiomechCoordinateSystem(
    #     antero_posterior_axis=CartesianAxis.plusX,
    #     infero_superior_axis=CartesianAxis.plusZ,
    #     medio_lateral_axis=CartesianAxis.minusY,
    #     segment=None,
    #     origin=None,
    # )
    # print(thorax.is_isb_oriented)
    # assert thorax.is_isb_oriented == False
    # clavicle = BiomechCoordinateSystem(
    #     antero_posterior_axis=CartesianAxis.plusX,
    #     infero_superior_axis=CartesianAxis.plusZ,
    #     medio_lateral_axis=CartesianAxis.minusY,
    #     segment=None,
    #     origin=None,
    # )
    # assert clavicle.is_isb_oriented == False
    # print(clavicle.is_isb_oriented)
    # sterno_clav = Joint(
    #     joint_type=JointType.STERNO_CLAVICULAR,
    #     euler_sequence=EulerSequence.ZXY,
    #     translation_frame=None,
    #     translation_origin=None,
    # )
    # print(sterno_clav.is_joint_sequence_isb)
    # assert sterno_clav.is_joint_sequence_isb == False


def test_joint_sequence_isb():
    print("todo")
    # joint = Joint(JointType.STERNO_CLAVICULAR, EulerSequence.YXZ, AnatomicalLandmark.Thorax.T7, Segment.THORAX)
    # assert joint.is_joint_sequence_isb == True


def test_isb_euler_sequence():
    print("todo")
    # joint = Joint(
    #     JointType.GLENO_HUMERAL, EulerSequence.YXY, AnatomicalLandmark.Humerus.GLENOHUMERAL_HEAD, Segment.HUMERUS
    # )
    # assert joint.isb_euler_sequence() == EulerSequence.YXY


def test_is_sequence_convertible_through_factors():
    print("todo")
    # joint = Joint(JointType.STERNO_CLAVICULAR, EulerSequence.YXZ, AnatomicalLandmark.Thorax.T7, Segment.THORAX)
    # assert joint.is_sequence_convertible_through_factors() == True
    #
    # joint = Joint(
    #     JointType.GLENO_HUMERAL, EulerSequence.YXY, AnatomicalLandmark.Humerus.GLENOHUMERAL_HEAD, Segment.HUMERUS
    # )
    # assert joint.is_sequence_convertible_through_factors() == True
    #
    # joint = Joint(
    #     JointType.GLENO_HUMERAL, EulerSequence.YXZ, AnatomicalLandmark.Humerus.GLENOHUMERAL_HEAD, Segment.HUMERUS
    # )
    # assert joint.is_sequence_convertible_through_factors() == False


def test_biomech_origin_from_string():
    assert AnatomicalLandmark.from_string("T7") == AnatomicalLandmark.Thorax.T7
    assert AnatomicalLandmark.from_string("GH") == AnatomicalLandmark.Humerus.GLENOHUMERAL_HEAD
    assert AnatomicalLandmark.from_string("AC") == AnatomicalLandmark.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER

    with pytest.raises(ValueError):
        AnatomicalLandmark.from_string("INVALID_ORIGIN")


def test_segment_from_string():
    assert Segment.from_string("thorax") == Segment.THORAX
    assert Segment.from_string("humerus") == Segment.HUMERUS

    with pytest.raises(ValueError):
        Segment.from_string("INVALID_SEGMENT")
