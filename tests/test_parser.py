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
    assert scapula_frame.is_isb is False

    assert scapula_frame.is_x_axis_postero_anterior is False
    assert scapula_frame.is_y_axis_supero_inferior is False
    assert scapula_frame.is_z_axis_medio_lateral is False

    assert scapula_frame.postero_anterior_axis == scapula_frame.y_axis
    assert scapula_frame.infero_superior_axis == scapula_frame.z_axis
    assert scapula_frame.medio_lateral_axis == scapula_frame.x_axis

    assert scapula_frame.medio_lateral_local_axis == CartesianAxis.minusX
    assert scapula_frame.infero_superior_local_axis == CartesianAxis.plusZ
    assert scapula_frame.postero_anterior_local_axis == CartesianAxis.minusY


def isb_verification(frame: Frame):
    assert frame.is_x_axis_postero_anterior is True
    assert frame.is_y_axis_supero_inferior is True
    assert frame.is_z_axis_medio_lateral is True

    assert frame.postero_anterior_local_axis == CartesianAxis.plusX
    assert frame.infero_superior_local_axis == CartesianAxis.plusY
    assert frame.medio_lateral_local_axis == CartesianAxis.plusZ

    assert frame.postero_anterior_axis == frame.x_axis
    assert frame.infero_superior_axis == frame.y_axis
    assert frame.medio_lateral_axis == frame.z_axis

    print(frame.origin)
    assert frame.is_origin_isb is True
    assert frame.has_isb_landmarks is True
    assert frame.is_isb is True


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
    isb_verification(isb_scapula_frame)


thorax_frame_template = lambda axis: Frame.from_xyz_string(
    x_axis="y^z", y_axis="vec((T8+PX)/2>(C7+IJ)/2)", z_axis=axis, origin="IJ", segment=Segment.THORAX
)
combinations_for_z_axis = [
    "vec((PX+T8)/2>IJ)^vec((PX+T8)/2>C7)",
    "vec(C7>(PX+T8)/2)^vec(C7>IJ)",
    "vec(IJ>C7)^vec(IJ>(PX+T8)/2)",
]

thorax_frames_isb = [thorax_frame_template(axis) for axis in combinations_for_z_axis]


@pytest.mark.parametrize(
    "isb_thorax_frame",
    thorax_frames_isb,
)
def test_parse_thorax_frame_isb(isb_thorax_frame):
    isb_verification(isb_thorax_frame)


def test_thorax_not_isb():
    # e.g. : Moissenet
    thorax_frame = Frame.from_xyz_string(
        x_axis="y^z",
        y_axis="vec((T8+PX)/2>(C7+IJ)/2)",
        z_axis="vec(T8>IJ)^vec(T8>C7)",  # because of this line, it is not ISB
        origin="IJ",
        segment=Segment.THORAX,
    )

    assert thorax_frame.is_isb == False
    assert thorax_frame.has_isb_landmarks == False
    assert thorax_frame.is_origin_isb == True

    assert thorax_frame.is_x_axis_postero_anterior == True
    assert thorax_frame.is_y_axis_supero_inferior == True
    assert thorax_frame.is_z_axis_medio_lateral == True

    assert thorax_frame.postero_anterior_axis == thorax_frame.x_axis
    assert thorax_frame.infero_superior_axis == thorax_frame.y_axis
    assert thorax_frame.medio_lateral_axis == thorax_frame.z_axis

    assert thorax_frame.postero_anterior_local_axis == CartesianAxis.plusX
    assert thorax_frame.infero_superior_local_axis == CartesianAxis.plusY
    assert thorax_frame.medio_lateral_local_axis == CartesianAxis.plusZ


def test_isb_clavicle():
    y_thorax = f"vec((T8+PX)/2>(C7+IJ)/2)"
    clavicle_frame = Frame.from_xyz_string(
        x_axis=f"{y_thorax}^z", y_axis="z^x", z_axis="vec(SC>AC)", origin="SC", segment=Segment.CLAVICLE
    )

    isb_verification(clavicle_frame)


def test_not_isb_clavicle():
    y_thorax = f"vec(T8>(C7+IJ)/2)"
    clavicle_frame = Frame.from_xyz_string(
        x_axis=f"{y_thorax}^z", y_axis="z^x", z_axis="vec(SC>AC)", origin="SC", segment=Segment.CLAVICLE
    )

    assert clavicle_frame.is_x_axis_postero_anterior == True
    assert clavicle_frame.is_y_axis_supero_inferior == True
    assert clavicle_frame.is_z_axis_medio_lateral == True

    assert clavicle_frame.postero_anterior_axis == clavicle_frame.x_axis
    assert clavicle_frame.infero_superior_axis == clavicle_frame.y_axis
    assert clavicle_frame.medio_lateral_axis == clavicle_frame.z_axis

    assert clavicle_frame.postero_anterior_local_axis == CartesianAxis.plusX
    assert clavicle_frame.infero_superior_local_axis == CartesianAxis.plusY
    assert clavicle_frame.medio_lateral_local_axis == CartesianAxis.plusZ

    assert clavicle_frame.is_origin_isb == True
    assert clavicle_frame.has_isb_landmarks == False
    assert clavicle_frame.is_isb == False

    z_thorax = f"vec((T8+PX)/2>(C7+IJ)/2)"
    clavicle_frame = Frame.from_xyz_string(
        x_axis="vec(SC>AC)", y_axis=f"{z_thorax}^x", z_axis="x^y", origin="SC", segment=Segment.CLAVICLE
    )

    assert clavicle_frame.is_x_axis_postero_anterior == False
    assert clavicle_frame.is_y_axis_supero_inferior == False
    assert clavicle_frame.is_z_axis_medio_lateral == False

    assert clavicle_frame.postero_anterior_axis == clavicle_frame.y_axis
    assert clavicle_frame.infero_superior_axis == clavicle_frame.z_axis
    assert clavicle_frame.medio_lateral_axis == clavicle_frame.x_axis

    assert clavicle_frame.is_origin_isb == True
    assert clavicle_frame.has_isb_landmarks == True
    assert clavicle_frame.is_isb == False


humerus_frame_template = lambda x_axis: Frame.from_xyz_string(
    x_axis=x_axis, y_axis="vec((EM+EL)/2>GH)", z_axis="x^y", origin="GH", segment=Segment.HUMERUS
)
combinations_for_x_axis = [
    "vec(EL>EM)^vec(EL>GH)",
    "vec(GH>EL)^vec(GH>EM)",
    "vec(EM>GH)^vec(EM>EL)",
]

humerus_frames_isb = [humerus_frame_template(axis) for axis in combinations_for_x_axis]


@pytest.mark.parametrize(
    "isb_humerus_frame",
    humerus_frames_isb,
)
def test_parse_humerus_frame_isb(isb_humerus_frame):
    isb_verification(isb_humerus_frame)
