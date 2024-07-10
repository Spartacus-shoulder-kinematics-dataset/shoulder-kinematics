from .biomech_system import BiomechCoordinateSystem
from .checks import check_segment_filled_with_nan
from .enums_biomech import Segment, JointType, EulerSequence, AnatomicalLandmark, FrameType
from .frame_reader import Frame
from .joint import Joint
from .thoracohumeral_angle import ThoracohumeralAngle
from .utils import get_segment_columns_direction, get_segment_columns


def set_parent_segment_from_row(row, segment: Segment):
    segment_cols_direction = get_segment_columns_direction(segment)
    method = (
        Frame.from_global_thorax_strings
        if segment == Segment.THORAX and row.thorax_is_global
        else Frame.from_xyz_string
    )
    frame_parent = method(
        x_axis=row[segment_cols_direction[0]],
        y_axis=row[segment_cols_direction[1]],
        z_axis=row[segment_cols_direction[2]],
        origin=row[segment_cols_direction[3]],
        segment=segment,
        side="right" if row.side_as_right or segment == Segment.THORAX else row.side,
    )
    parent_biomech_sys = BiomechCoordinateSystem.from_frame(frame_parent)
    return parent_biomech_sys


def set_child_segment_from_row(row, segment: Segment):
    segment_cols_direction = get_segment_columns_direction(segment)
    frame_child = Frame.from_xyz_string(
        x_axis=row[segment_cols_direction[0]],
        y_axis=row[segment_cols_direction[1]],
        z_axis=row[segment_cols_direction[2]],
        origin=row[segment_cols_direction[3]],
        segment=segment,
        side="right" if row.side_as_right else row.side,
    )

    if frame_child.only_translation:
        # for Nishinaka for example
        child_biomech_sys = BiomechCoordinateSystem(
            antero_posterior_axis=None,
            infero_superior_axis=None,
            medio_lateral_axis=None,
            origin=frame_child.origin,
            segment=segment,
            frame=frame_child,
        )
    else:
        child_biomech_sys = BiomechCoordinateSystem.from_frame(frame_child)

    return child_biomech_sys


def set_joint_from_row(row, joint: JointType):

    try:
        translation_frame = FrameType.from_string(row.displacement_cs)
    except:
        translation_frame = None

    return Joint(
        joint_type=JointType.from_string(row.joint),
        euler_sequence=EulerSequence.from_string(row.euler_sequence),  # throw a None
        translation_origin=AnatomicalLandmark.from_string(row.origin_displacement),
        translation_frame=translation_frame,
        parent_segment=set_parent_segment_from_row(row, joint.parent),
        child_segment=set_child_segment_from_row(row, joint.child),
    )


def set_thoracohumeral_angle_from_row(row):

    # some data are very sparse, so we need to check if the humerus segment is filled
    has_no_humerus_info = check_segment_filled_with_nan(row, get_segment_columns(Segment.HUMERUS))

    return ThoracohumeralAngle(
        euler_sequence=EulerSequence.from_string(row.thoracohumeral_sequence),
        angle=row.thoracohumeral_angle,
        parent_segment=set_parent_segment_from_row(row, Segment.THORAX),
        child_segment=None if has_no_humerus_info else set_child_segment_from_row(row, Segment.HUMERUS),
    )
