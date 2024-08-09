import pytest

from spartacus import (
    AnatomicalLandmark,
    Segment,
)


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
