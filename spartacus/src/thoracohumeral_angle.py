import numpy as np

from .biomech_system import BiomechCoordinateSystem
from .corrections.angle_conversion_callbacks import convert_euler_angles_and_frames_to_isb
from .enums_biomech import EulerSequence, CartesianAxis, JointType


class ThoracohumeralAngle:
    def __init__(
        self,
        euler_sequence: EulerSequence,
        angle: CartesianAxis | str,
        parent_segment: BiomechCoordinateSystem,
        child_segment: BiomechCoordinateSystem,
    ):

        self.euler_sequence = euler_sequence
        self.angle = angle
        self.joint_type = JointType.THORACO_HUMERAL
        self.parent_segment = parent_segment
        self.child_segment = child_segment

    @property
    def is_euler_sequence_isb(self) -> bool:
        return EulerSequence.isb_from_joint_type(self.joint_type) == self.euler_sequence

    @property
    def is_elevation_angle_isb(self) -> bool:
        if self.is_euler_sequence_equivalent_to_isb:
            return self.angle == "x"

        return False

    @property
    def isb_euler_sequence(self) -> EulerSequence:
        return EulerSequence.isb_from_joint_type(self.joint_type)

    @property
    def is_euler_sequence_equivalent_to_isb(self) -> bool:

        if self.is_euler_sequence_isb:
            return True

        value_rot1 = 0.2
        value_rot2 = 0.3
        value_rot3 = 0.4

        new_angles = convert_euler_angles_and_frames_to_isb(
            previous_sequence_str=self.euler_sequence.value,
            new_sequence_str=EulerSequence.isb_from_joint_type(self.joint.joint_type).value,
            rot1=value_rot1,
            rot2=value_rot2,
            rot3=value_rot3,
            bsys_parent=self.parent_segment,
            bsys_child=self.child_segment,
        )

        is_first_angle_equal = np.round(new_angles[0], 5) == value_rot1
        is_second_angle_equal = np.round(new_angles[1], 5) == value_rot2
        is_third_angle_equal = np.round(new_angles[2], 5) == value_rot3

        if is_first_angle_equal and is_second_angle_equal and is_third_angle_equal:
            return True

        return False
