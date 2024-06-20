import numpy as np

from .enums_biomech import CartesianAxis, BiomechDirection, AnatomicalLandmark, Segment
from .frame_reader import Frame
from .utils import compute_rotation_matrix_from_axes


class BiomechCoordinateSystem:
    def __init__(
        self,
        segment: Segment,
        antero_posterior_axis: CartesianAxis,
        infero_superior_axis: CartesianAxis,
        medio_lateral_axis: CartesianAxis,
        origin=None,
        frame: Frame = None,
    ):
        # verify isinstance
        if not isinstance(antero_posterior_axis, CartesianAxis):
            raise TypeError("antero_posterior_axis should be of type CartesianAxis")
        if not isinstance(infero_superior_axis, CartesianAxis):
            raise TypeError("infero_superior_axis should be of type CartesianAxis")
        if not isinstance(medio_lateral_axis, CartesianAxis):
            raise TypeError("medio_lateral_axis should be of type CartesianAxis")
        # verity they are all different
        if (
            antero_posterior_axis == infero_superior_axis
            or antero_posterior_axis == medio_lateral_axis
            or infero_superior_axis == medio_lateral_axis
        ):
            raise ValueError("antero_posterior_axis, infero_superior_axis, medio_lateral_axis should be different")

        self.anterior_posterior_axis = antero_posterior_axis
        self.infero_superior_axis = infero_superior_axis
        self.medio_lateral_axis = medio_lateral_axis

        self.origin = origin
        self.segment = segment
        self.frame = frame

    @classmethod
    def from_biomech_directions(
        cls,
        x: BiomechDirection,
        y: BiomechDirection,
        z: BiomechDirection,
        origin: AnatomicalLandmark = None,
        segment: Segment = None,
    ):
        my_arg = dict()

        # verify each of the x, y, z is different
        if x == y or x == z or y == z:
            raise ValueError("x, y, z should be different")

        # verify is positive or negative
        actual_axes = [x, y, z]

        axis_to_key = {
            BiomechDirection.PlusPosteroAnterior: "antero_posterior_axis",
            BiomechDirection.MinusPosteroAnterior: "antero_posterior_axis",
            BiomechDirection.PlusMedioLateral: "medio_lateral_axis",
            BiomechDirection.MinusMedioLateral: "medio_lateral_axis",
            BiomechDirection.PlusInferoSuperior: "infero_superior_axis",
            BiomechDirection.MinusInferoSuperior: "infero_superior_axis",
        }

        sign_to_cartesian_axis_x = {
            1: CartesianAxis.plusX,
            -1: CartesianAxis.minusX,
        }
        sign_to_cartesian_axis_y = {
            1: CartesianAxis.plusY,
            -1: CartesianAxis.minusY,
        }
        sign_to_cartesian_axis_z = {
            1: CartesianAxis.plusZ,
            -1: CartesianAxis.minusZ,
        }

        for axis, sign_to_cartesian_axis in zip(
            actual_axes, [sign_to_cartesian_axis_x, sign_to_cartesian_axis_y, sign_to_cartesian_axis_z]
        ):
            my_arg[axis_to_key[axis]] = sign_to_cartesian_axis[axis.sign]

        my_arg["origin"] = origin
        my_arg["segment"] = segment

        return cls(**my_arg)

    @classmethod
    def from_frame(cls, frame: Frame):
        return cls(
            segment=frame.segment,
            antero_posterior_axis=frame.postero_anterior_local_axis,
            infero_superior_axis=frame.infero_superior_local_axis,
            medio_lateral_axis=frame.medio_lateral_local_axis,
            origin=frame.origin,
            frame=frame,
        )

    @property
    def is_isb_oriented(self) -> bool:
        condition_1 = self.anterior_posterior_axis is CartesianAxis.plusX
        condition_2 = self.infero_superior_axis is CartesianAxis.plusY
        condition_3 = self.medio_lateral_axis is CartesianAxis.plusZ
        return condition_1 and condition_2 and condition_3

    @property
    def is_isb_origin(self) -> bool:
        segment_to_origin_isb = {
            Segment.SCAPULA: AnatomicalLandmark.Scapula.origin_isb,
            Segment.THORAX: AnatomicalLandmark.Thorax.origin_isb,
            Segment.CLAVICLE: AnatomicalLandmark.Clavicle.origin_isb,
            Segment.HUMERUS: AnatomicalLandmark.Humerus.origin_isb,
        }

        return segment_to_origin_isb.get(self.segment) == self.origin

    def is_origin_on_an_isb_axis(self) -> bool:
        """
        Return True if the origin is on an ISB axis, False otherwise

        NOTE
        ----
        The true definition would be, the origin is part of the process to build an ISB axis.

        """
        if self.is_isb_origin:
            return True
        # todo: may check according to frame object
        ON_ISB_AXES = {
            Segment.THORAX: [AnatomicalLandmark.Thorax.C7, AnatomicalLandmark.Thorax.T8, AnatomicalLandmark.Thorax.PX],
            Segment.CLAVICLE: [
                AnatomicalLandmark.Clavicle.STERNOCLAVICULAR_JOINT_CENTER,
                AnatomicalLandmark.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER,
            ],
            Segment.SCAPULA: [AnatomicalLandmark.Scapula.TRIGNONUM_SPINAE, AnatomicalLandmark.Scapula.ANGULUS_INFERIOR],
            Segment.HUMERUS: [AnatomicalLandmark.Humerus.MIDPOINT_EPICONDYLES],
        }

        return self.origin in ON_ISB_AXES.get(self.segment, [])

    def is_isb(self) -> bool:
        # excluding thorax is global
        return self.frame.is_isb if self.frame is not None else False

    def is_direct(self) -> bool:
        """check if the frame is direct (True) or indirect (False)"""
        return np.linalg.det(self.get_rotation_matrix()) > 0

    def get_rotation_matrix(self):
        """
        write the rotation matrix from the cartesian axis

        such that a_in_isb = R_to_isb_from_local @ a_in_local

        """
        # todo: to transfer in Frame ?
        return compute_rotation_matrix_from_axes(
            anterior_posterior_axis=self.anterior_posterior_axis.value[1][:, np.newaxis],
            infero_superior_axis=self.infero_superior_axis.value[1][:, np.newaxis],
            medio_lateral_axis=self.medio_lateral_axis.value[1][:, np.newaxis],
        )

    # def get_segment_risk_quantification(self, type_risk):
    #     """
    #     Return the risk quantification of the segment which is the product of the risk
    #     of each type of risk described in the dictionnary dict_coeff.
    #
    #     Parameters
    #     ----------
    #     type_risk: str
    #         "rotation" or "displacement"
    #     """
    #
    #     risk = 1
    #     if self.is_mislabeled():
    #         risk = risk * DEVIATION_COEFF[type_risk]["label"]
    #
    #     if not self.is_isb_origin():
    #         risk = risk * DEVIATION_COEFF[type_risk]["origin"]
    #
    #     if self.is_any_axis_wrong_sens():
    #         risk = risk * DEVIATION_COEFF[type_risk]["sens"]

    # return risk

    def __print__(self):
        print(f"Segment: {self.segment}")
        print(f"Origin: {self.origin}")
        print(f"Anterior Posterior Axis: {self.anterior_posterior_axis}")
        print(f"Medio Lateral Axis: {self.medio_lateral_axis}")
        print(f"Infero Superior Axis: {self.infero_superior_axis}")
