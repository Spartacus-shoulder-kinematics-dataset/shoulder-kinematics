from .enums_biomech import EulerSequence, JointType, Segment, AnatomicalLandmark
from .legend_utils import isb_rotation_biomechanical_dof


class Joint:
    def __init__(
        self,
        joint_type: JointType,
        euler_sequence: EulerSequence,
        translation_origin: AnatomicalLandmark,
        translation_frame: Segment,
    ):
        self.joint_type = joint_type
        self.euler_sequence = euler_sequence
        self.translation_origin = translation_origin
        self.translation_frame = translation_frame

    def is_joint_sequence_isb(self) -> bool:
        return EulerSequence.isb_from_joint_type(self.joint_type) == self.euler_sequence

    def isb_euler_sequence(self) -> EulerSequence:
        return EulerSequence.isb_from_joint_type(self.joint_type)

    # todo: stuff for translations ..?

    def is_sequence_convertible_through_factors(self, print_warning: bool = False) -> bool:
        """
        Check if the euler sequence of the joint can be converted to the ISB sequence with factors 1 or -1

        We expect the euler sequence to have three different letters, if the ISB sequence is YXZ (steroclavicular, acromioclavicular, scapulothoracic)
        We expect the euler sequence to have two different letters, if the ISB sequence is YXY (glenohumeral, thoracohumeral)

        Return
        ------
        bool
            True if the sequence can be converted with factors 1 or -1, False otherwise
        """
        if self.joint_type in (JointType.STERNO_CLAVICULAR, JointType.ACROMIO_CLAVICULAR, JointType.SCAPULO_THORACIC):
            sequence_wanted = EulerSequence.YXZ
            # check that we have three different letters in the sequence
            if len(set(self.euler_sequence.value)) != 3:
                if print_warning:
                    print(
                        "The euler sequence of the joint must have three different letters to be able to convert with factors 1"
                        f"or -1 to the ISB sequence {sequence_wanted.value}, but the sequence of the joint is"
                        f" {self.euler_sequence.value}"
                    )
                return False

        elif self.joint_type in (JointType.GLENO_HUMERAL, JointType.THORACO_HUMERAL):
            sequence_wanted = EulerSequence.YXY
            # check that the sequence in joint.euler_sequence as the same two letters for the first and third rotations
            if self.euler_sequence.value[0] != self.euler_sequence.value[2]:
                if print_warning:
                    print(
                        "The euler sequence of the joint must have the same two letters for the first and third rotations"
                        f"to be able to convert with factors 1 or -1 to the ISB sequence {sequence_wanted.value},"
                        f" but the sequence of the joint is {self.euler_sequence.value}"
                    )
                return False
        else:
            if print_warning:
                print(
                    "The joint type must be JointType.STERNO_CLAVICULAR, JointType.ACROMIO_CLAVICULAR,"
                    "JointType.SCAPULO_THORACIC, JointType.GLENO_HUMERAL, JointType.THORACO_HUMERAL"
                )
            return False

        return True

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
