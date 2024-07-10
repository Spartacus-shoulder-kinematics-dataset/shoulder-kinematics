from .biomech_system import BiomechCoordinateSystem
from .deviation_constant import DEVIATION_COEFF
from .enums_biomech import CartesianAxis
from .joint import Joint
from .thoracohumeral_angle import ThoracohumeralAngle


class Compliance:

    def __init__(self, mode: str):
        self.mode = mode

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

    def isb_euler_sequence(self, joint: Joint) -> float:
        if Compliance.is_euler_sequence_equivalent_to_isb(joint):
            return 1

        return DEVIATION_COEFF[self.mode]["euler_sequence"]

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

    def translation_frame_proximal_isb(self, joint: Joint) -> float:
        if Compliance.is_translation_frame_proximal_isb(joint):
            return 1

        return DEVIATION_COEFF[self.mode]["translation_frame"]

    @staticmethod
    def is_thoraco_humeral_angle_isb(thoraco_humeral_angle: ThoracohumeralAngle) -> bool:
        """
        Check if the humerothoracic angle is computed from the Euler angles.

        Parameters
        ----------
        thoraco_humeral_angle: ThoracohumeralAngle
        """
        return thoraco_humeral_angle.is_elevation_angle_isb

    def thoraco_humeral_angle_isb(self, thoraco_humeral_angle: ThoracohumeralAngle) -> float:
        if Compliance.is_thoraco_humeral_angle_isb(thoraco_humeral_angle):
            return 1

        return DEVIATION_COEFF[self.mode]["translation_frame"]


class SegmentCompliance(Compliance):
    def __init__(self, mode: str, bsys: BiomechCoordinateSystem):
        super().__init__(mode)
        self.bsys = bsys

    @property
    def is_c1(self) -> bool:
        return not self.is_isb_oriented(self.bsys)

    @property
    def is_c2(self) -> bool:
        return not self.are_axes_built_with_isb_landmarks(self.bsys)

    @property
    def is_c3(self) -> bool:
        return not self.is_origin_isb(self.bsys)


class JointCompliance(Compliance):
    def __init__(self, mode: str, joint: Joint, thoracohumeral_angle: ThoracohumeralAngle):
        super().__init__(mode)
        self.joint = joint
        self.thoracohumeral_angle = thoracohumeral_angle

    @property
    def is_c4(self) -> bool:
        return not self.is_euler_sequence_equivalent_to_isb(self.joint)

    @property
    def is_c5(self) -> bool:
        return not self.is_translation_frame_proximal_isb(self.joint)

    @property
    def is_c6(self) -> bool:
        return not self.is_thoraco_humeral_angle_isb(self.thoracohumeral_angle)
