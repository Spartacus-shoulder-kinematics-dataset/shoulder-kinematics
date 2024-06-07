import pytest

from spartacus.src.enums_biomech import AnatomicalLandmark, Segment, CartesianAxis
from spartacus.src.frame_reader import parse_axis, Frame


def test_parse_landmarks():
    test_cases = [
        {"input": "vec(T10>PX)", "expected": "Start: Thorax.T10, End: Thorax.PX"},
        {
            "input": "vec(T10>PX)^vec((T10+PX)/2>(IJ+T1)/2)",
            "expected": "(Start: Thorax.T10, End: Thorax.PX) "
            "X (Start: Thorax.MIDPOINT_T10_PX, End: Thorax.MIDPOINT_IJ_T1)",
        },
        {
            "input": "vec((EL+EM)/2>GH)^vec(EL>EM)",
            "expected": "(Start: Humerus.MIDPOINT_EPICONDYLES, End: Humerus.GLENOHUMERAL_HEAD) "
            "X (Start: Humerus.LATERAL_EPICONDYLE, End: Humerus.MEDIAL_EPICONDYLE)",
        },
        {
            "input": "vec(EM>GH)^vec(EL>EM)",
            "expected": "(Start: Humerus.MEDIAL_EPICONDYLE, End: Humerus.GLENOHUMERAL_HEAD) "
            "X (Start: Humerus.LATERAL_EPICONDYLE, End: Humerus.MEDIAL_EPICONDYLE)",
        },
    ]

    for case in test_cases:
        input_str = case["input"]
        expected_output = case["expected"]
        actual_output = parse_axis(input_str)
        assert str(actual_output) == expected_output


def test_parse_frame():
    scapula_frame = Frame.from_xyz_string(
        x_axis="vec(AA>TS)", y_axis="vec(IA>AA)^vec(AA>TS)", z_axis="x^y", origin="AC", segment=Segment.SCAPULA
    )
    assert scapula_frame.origin == AnatomicalLandmark.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER
    assert str(scapula_frame.x_axis) == "Start: Scapula.ANGULAR_ACROMIALIS, End: Scapula.TRIGNONUM_SPINAE"
    assert (
        str(scapula_frame.y_axis) == "(Start: Scapula.ANGULUS_INFERIOR, End: Scapula.ANGULAR_ACROMIALIS) "
        "X (Start: Scapula.ANGULAR_ACROMIALIS, End: Scapula.TRIGNONUM_SPINAE)"
    )
    assert (
        str(scapula_frame.z_axis) == "(Start: Scapula.ANGULAR_ACROMIALIS, End: Scapula.TRIGNONUM_SPINAE) X "
        "("
        "(Start: Scapula.ANGULUS_INFERIOR, End: Scapula.ANGULAR_ACROMIALIS) X "
        "(Start: Scapula.ANGULAR_ACROMIALIS, End: Scapula.TRIGNONUM_SPINAE)"
        ")"
    )
    assert scapula_frame.landmarks == (
        AnatomicalLandmark.Scapula.ANGULAR_ACROMIALIS,
        AnatomicalLandmark.Scapula.TRIGNONUM_SPINAE,
        AnatomicalLandmark.Scapula.ANGULUS_INFERIOR,
    )
    assert scapula_frame.is_isb == False

    assert scapula_frame.is_x_axis_postero_anterior == False
    assert scapula_frame.is_y_axis_supero_inferior == False
    assert scapula_frame.is_z_axis_medio_lateral == False

    assert scapula_frame.postero_anterior_axis == scapula_frame.y_axis
    assert scapula_frame.infero_superior_axis == scapula_frame.z_axis
    assert scapula_frame.medio_lateral_axis == scapula_frame.x_axis

    assert scapula_frame.medio_lateral_local_axis == CartesianAxis.minusZ
    assert scapula_frame.infero_superior_local_axis == CartesianAxis.plusY
    assert scapula_frame.postero_anterior_local_axis == CartesianAxis.minusX


scapula_frame_template = lambda x_axis: Frame.from_xyz_string(
    z_axis="vec(TS>AA)", x_axis=x_axis, y_axis="z^x", origin="AA", segment=Segment.SCAPULA
)
combinations_for_x_axis = ["vec(TS>AA)^vec(TS>IA)", "vec(AA>IA)^vec(AA>TS)", "vec(IA>TS)^vec(IA>AA)"]

scapula_frame_isb = [scapula_frame_template(x_axis) for x_axis in combinations_for_x_axis]


@pytest.mark.parametrize(
    "isb_scapula_frame",
    scapula_frame_isb,
)
def test_parse_scapula_frame_isb(isb_scapula_frame):
    assert isb_scapula_frame.is_x_axis_postero_anterior == True
    assert isb_scapula_frame.is_y_axis_supero_inferior == True
    assert isb_scapula_frame.is_z_axis_medio_lateral == True

    assert isb_scapula_frame.postero_anterior_local_axis == CartesianAxis.plusX
    assert isb_scapula_frame.infero_superior_local_axis == CartesianAxis.plusY
    assert isb_scapula_frame.medio_lateral_local_axis == CartesianAxis.plusZ

    assert isb_scapula_frame.postero_anterior_axis == isb_scapula_frame.x_axis
    assert isb_scapula_frame.infero_superior_axis == isb_scapula_frame.y_axis
    assert isb_scapula_frame.medio_lateral_axis == isb_scapula_frame.z_axis

    print(isb_scapula_frame.origin)
    assert isb_scapula_frame.is_origin_isb == True
    assert isb_scapula_frame.has_isb_landmarks == True
    assert isb_scapula_frame.is_isb == True
