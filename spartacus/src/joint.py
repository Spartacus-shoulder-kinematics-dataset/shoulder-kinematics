import numpy as np

from .biomech_system import BiomechCoordinateSystem
from .corrections.angle_conversion_callbacks import convert_euler_angles_and_frames_to_isb
from .enums_biomech import EulerSequence, JointType, AnatomicalLandmark, FrameType
from .legend_utils import isb_rotation_biomechanical_dof


class Joint:
    """
    A class to represent a joint in a biomechanical system.

    Attributes
    ----------
    joint_type : JointType
        The type of the joint.
    euler_sequence : EulerSequence
        The Euler sequence used for the joint.
    translation_origin : AnatomicalLandmark
        The anatomical landmark used as the origin for translation.
    translation_frame : FrameType
        The frame type used for translation.
    parent_segment : BiomechCoordinateSystem
        The parent biomechanical coordinate system.
    child_segment : BiomechCoordinateSystem
        The child biomechanical coordinate system.
    """

    def __init__(
        self,
        joint_type: JointType,
        euler_sequence: EulerSequence,
        translation_origin: AnatomicalLandmark,
        translation_frame: FrameType,
        parent_segment: BiomechCoordinateSystem,
        child_segment: BiomechCoordinateSystem,
    ):
        """
        Constructs all the necessary attributes for the Joint object.

        Parameters
        ----------
        joint_type : JointType
            The type of the joint.
        euler_sequence : EulerSequence
            The Euler sequence used for the joint.
        translation_origin : AnatomicalLandmark
            The anatomical landmark used as the origin for translation.
        translation_frame : FrameType
            The frame type used for translation.
        parent_segment : BiomechCoordinateSystem
            The parent biomechanical coordinate system.
        child_segment : BiomechCoordinateSystem
            The child biomechanical coordinate system.
        """
        self.joint_type = joint_type
        self.euler_sequence = euler_sequence
        self.translation_origin = translation_origin
        self.translation_frame = translation_frame
        self.parent_segment = parent_segment
        self.child_segment = child_segment

    @property
    def is_joint_sequence_isb(self) -> bool:
        return EulerSequence.isb_from_joint_type(self.joint_type) == self.euler_sequence

    @property
    def isb_euler_sequence(self) -> EulerSequence:
        return EulerSequence.isb_from_joint_type(self.joint_type)

    @property
    def is_euler_sequence_equivalent_to_isb(self) -> bool:
        if self.euler_sequence == self.isb_euler_sequence:
            return True
        if self.euler_sequence is None:
            return False

        value_rot1 = 0.2
        value_rot2 = 0.3
        value_rot3 = 0.4

        new_angles = convert_euler_angles_and_frames_to_isb(
            previous_sequence_str=self.euler_sequence.value,
            new_sequence_str=EulerSequence.isb_from_joint_type(self.joint_type).to_string,
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

    @property
    def is_translation_frame_proximal_isb(self) -> bool:
        frame_type_dict = {
            FrameType.CHILD: False,
            FrameType.JCS: False,
            FrameType.PARENT: True,
        }

        return frame_type_dict.get(self.translation_frame)

    @property
    def isb_rotation_biomechanical_dof(self) -> (str, str, str):
        return isb_rotation_biomechanical_dof(self.joint_type)

    @property
    def isb_translation_biomechanical_dof(self) -> (str, str, str):
        # TODO : Put correct dof
        raise NotImplementedError("Not implemented yet")

        joint_mapping = {
            JointType.GLENO_HUMERAL: ("flexion_extension", "abduction_adduction", "internal_external_rotation"),
            JointType.SCAPULO_THORACIC: (
                "upward_downward_rotation",
                "anterior_posterior_tilt",
                "internal_external_rotation",
            ),
            JointType.ACROMIO_CLAVICULAR: (
                "anterior_posterior_tilt",
                "internal_external_rotation",
                "upward_downward_rotation",
            ),
            JointType.STERNO_CLAVICULAR: (
                "anterior_posterior_tilt",
                "internal_external_rotation",
                "upward_downward_rotation",
            ),
            JointType.THORACO_HUMERAL: ("flexion_extension", "abduction_adduction", "internal_external_rotation"),
        }

        return joint_mapping.get(self.joint_type)
