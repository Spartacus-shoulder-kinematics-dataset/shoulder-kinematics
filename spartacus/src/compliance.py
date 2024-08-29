from .biomech_system import BiomechCoordinateSystem
from .enums_biomech import CartesianAxis
from .joint import Joint
from .thoracohumeral_angle import ThoracohumeralAngle


class Compliance:
    """
    A class to represent compliance with biomechanical standards.

    Attributes
    ----------
    standards : str
        The standards to which the compliance is checked. Default is "ISB".
    """

    def __init__(self):
        self.standards = "ISB"  # todo: we could some day modify to get different standards requirements.

    @staticmethod
    def are_axes_isb_labeled(bsys: BiomechCoordinateSystem) -> bool:
        """
        Check if the axes are labeled according to the ISB recommendations.
        Return True if the segment is mislabeled, False otherwise
        Mislabeling is defined as the only difference with ISB being the wrong name of the axis. Which means that :
            - the antero posterior axis is not along the x axis
            - the infero superior axis is not along the y axis
            - the medio lateral axis is not along the z axis

        Parameters
        ----------
        bsys: BiomechCoordinateSystem
            The biomechanical coordinate system to check.
        """

        condition_1 = (bsys.anterior_posterior_axis is CartesianAxis.plusX) or (
            bsys.anterior_posterior_axis is CartesianAxis.minusX
        )
        condition_2 = (bsys.infero_superior_axis is CartesianAxis.plusY) or (
            bsys.infero_superior_axis is CartesianAxis.minusY
        )
        condition_3 = (bsys.medio_lateral_axis is CartesianAxis.plusZ) or (
            bsys.medio_lateral_axis is CartesianAxis.minusZ
        )

        return condition_1 and condition_2 and condition_3

    @staticmethod
    def are_axes_sign_correct(bsys: BiomechCoordinateSystem) -> bool:
        """
        Check if the axes are labeled with the correct sign.

        Parameters
        ----------
        bsys: BiomechCoordinateSystem
            The biomechanical coordinate system to check.

        Return True if any of the axis is in the wrong sens, False otherwise
        The wrong sens is defined as the axis pointing in the positive direction (which here correspond to forward, to the right and up).
        """

        is_ant_post_positive = all(bsys.anterior_posterior_axis.value[1] >= 0)
        is_med_lat_positive = all(bsys.medio_lateral_axis.value[1] >= 0)
        is_inf_sup_positive = all(bsys.infero_superior_axis.value[1] >= 0)

        return is_ant_post_positive and is_med_lat_positive and is_inf_sup_positive

    @staticmethod
    def is_isb_oriented(bsys: BiomechCoordinateSystem) -> bool:
        """
        Check if the biomechanical coordinate system is oriented according to the ISB recommendations.

        Parameters
        ----------
        bsys: BiomechCoordinateSystem
            The biomechanical coordinate system to check.
        """
        return bsys.is_isb_oriented

    @staticmethod
    def are_axes_built_with_isb_landmarks(bsys: BiomechCoordinateSystem) -> bool:
        """
        Check if the biomechanical coordinate system is built with ISB landmarks.

        Parameters
        ----------
        bsys: BiomechCoordinateSystem
            The biomechanical coordinate system to check.
        """
        # if thorax is global
        if bsys.segment == "thorax" and bsys.frame is None:
            return False

        return bsys.frame.has_isb_landmarks

    @staticmethod
    def is_origin_isb(bsys: BiomechCoordinateSystem) -> bool:
        """
        Check if the origin of the biomechanical coordinate system is built with ISB landmarks.

        Parameters
        ----------
        bsys: BiomechCoordinateSystem
            The biomechanical coordinate system to check.
        """
        return bsys.is_isb_origin

    @staticmethod
    def is_euler_sequence_equivalent_to_isb(joint: Joint) -> bool:
        """
        Check if the Euler sequence is equivalent to the ISB recommendations.

        Parameters
        ----------
        joint: Joint
            The joint to check.
        """
        return joint.is_euler_sequence_equivalent_to_isb

    @staticmethod
    def is_translation_frame_proximal_isb(joint: Joint) -> bool:
        """
        Check if the translation frame is built with ISB landmarks.

        Parameters
        ----------
        joint: Joint
            The joint to check.
        """
        return joint.is_translation_frame_proximal_isb

    @staticmethod
    def is_thoraco_humeral_angle_isb(thoraco_humeral_angle: ThoracohumeralAngle) -> bool:
        """
        Check if the humerothoracic angle is computed from the Euler angles.

        Parameters
        ----------
        thoraco_humeral_angle: ThoracohumeralAngle
        """
        return thoraco_humeral_angle.is_elevation_angle_isb


class SegmentCompliance(Compliance):
    """
    A class to represent segment compliance with biomechanical standards.

    Attributes
    ----------
    bsys : BiomechCoordinateSystem
        The biomechanical coordinate system to check for compliance.
    """

    def __init__(self, bsys: BiomechCoordinateSystem):
        """
        Constructs all the necessary attributes for the SegmentCompliance object.

        Parameters
        ----------
        bsys : BiomechCoordinateSystem
            The biomechanical coordinate system to check for compliance.
        """
        super().__init__()
        self.bsys = bsys

    @property
    def is_c1(self) -> bool:
        """
        Check if the biomechanical coordinate system is oriented according to the ISB recommendations.

        Returns
        -------
        bool
            True if the coordinate system is ISB oriented, False otherwise.
        """
        return self.is_isb_oriented(self.bsys)

    @property
    def is_c2(self) -> bool:
        """
        Check if the biomechanical coordinate system is built with ISB landmarks.

        Returns
        -------
        bool
            True if the coordinate system is built with ISB landmarks, False otherwise.
        """
        return self.are_axes_built_with_isb_landmarks(self.bsys)

    @property
    def is_c3(self) -> bool:
        """
        Check if the origin of the biomechanical coordinate system is built with ISB landmarks.

        Returns
        -------
        bool
            True if the origin is built with ISB landmarks, False otherwise.
        """
        return self.is_origin_isb(self.bsys)


class JointCompliance(Compliance):
    """
    A class to represent joint compliance with biomechanical standards.

    Attributes
    ----------
    joint : Joint
        The joint to check for compliance.
    thoracohumeral_angle : ThoracohumeralAngle
        The thoracohumeral angle to check for compliance.
    """

    def __init__(self, joint: Joint, thoracohumeral_angle: ThoracohumeralAngle):
        """
        Constructs all the necessary attributes for the JointCompliance object.

        Parameters
        ----------
        joint : Joint
            The joint to check for compliance.
        thoracohumeral_angle : ThoracohumeralAngle
            The thoracohumeral angle to check for compliance.
        """
        super().__init__()
        self.joint = joint
        self.thoracohumeral_angle = thoracohumeral_angle

    @property
    def is_c4(self) -> bool:
        """
        Check if the Euler sequence is equivalent to the ISB recommendations.

        Returns
        -------
        bool
            True if the Euler sequence is ISB equivalent, False otherwise.
        """
        return self.is_euler_sequence_equivalent_to_isb(self.joint)

    @property
    def is_c5(self) -> bool:
        """
        Check if the translation frame is built with ISB landmarks.

        Returns
        -------
        bool
            True if the translation frame is ISB compliant, False otherwise.
        """
        return self.is_translation_frame_proximal_isb(self.joint)

    @property
    def is_c6(self) -> bool:
        """
        Check if the humerothoracic angle is computed from the Euler angles.

        Returns
        -------
        bool
            True if the humerothoracic angle is ISB compliant, False otherwise.
        """
        return self.is_thoraco_humeral_angle_isb(self.thoracohumeral_angle)


class TotalCompliance:
    """
    A class to represent the total compliance with biomechanical standards.

    Attributes
    ----------
    parent : SegmentCompliance
        The parent segment compliance.
    child : SegmentCompliance
        The child segment compliance.
    joint : JointCompliance
        The joint compliance.
    """

    def __init__(
        self,
        parent_compliance: SegmentCompliance,
        child_compliance: SegmentCompliance,
        joint_compliance: JointCompliance,
    ):
        """
        Constructs all the necessary attributes for the TotalCompliance object.

        Parameters
        ----------
        parent_compliance : SegmentCompliance
            The parent segment compliance.
        child_compliance : SegmentCompliance
            The child segment compliance.
        joint_compliance : JointCompliance
            The joint compliance.
        """
        self.parent = parent_compliance
        self.child = child_compliance
        self.joint = joint_compliance

    @property
    def rotation(self):
        """
        Calculate the total rotation compliance.

        Returns
        -------
        int
            The total rotation compliance score.
        """
        return (
            self.parent.is_c1
            + self.parent.is_c2
            + self.child.is_c1
            + self.child.is_c2
            + self.joint.is_c4
            + self.joint.is_c6
        )

    @property
    def translation(self):
        """
        Calculate the total translation compliance.

        Returns
        -------
        int
            The total translation compliance score.
        """
        return (
            self.parent.is_c1
            + self.parent.is_c2
            + self.parent.is_c3
            + self.child.is_c3
            + self.joint.is_c5
            + self.joint.is_c6
        )

    @property
    def is_rotation_isb(self):
        """
        Check if the total rotation compliance is ISB compliant.

        Returns
        -------
        bool
            True if the total rotation compliance is ISB compliant, False otherwise.
        """
        return self.rotation == 6

    @property
    def is_translation_isb(self):
        """
        Check if the total translation compliance is ISB compliant.

        Returns
        -------
        bool
            True if the total translation compliance is ISB compliant, False otherwise.
        """
        return self.rotation == 6
