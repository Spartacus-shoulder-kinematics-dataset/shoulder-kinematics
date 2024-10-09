from abc import ABC, abstractmethod

import numpy as np

from .biomech_constant import get_constant
from .enums_biomech import AnatomicalLandmark, Segment, CartesianAxis, BiomechDirection, AnatomicalVector


class VectorBase(ABC):
    @property
    @abstractmethod
    def landmarks(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def compute_default_vector(self) -> np.ndarray:
        pass

    def principal_direction(self) -> CartesianAxis:
        """Returns the principal direction of the vector, ex: np.array([0.8, 0.2, -0.5]) -> CartesianAxis.plusX"""
        return CartesianAxis.principal_axis(self.compute_default_vector())

    def biomech_direction(self) -> BiomechDirection:
        """Returns the biomechanical direction of the vector, ex: np.array([0.8, 0.2, -0.5]) -> BiomechDirection.postero_anterior"""
        return BiomechDirection.from_direction_global_isb_frame(self.principal_direction())


class SoloVector(VectorBase):
    def __init__(self, direction: AnatomicalVector, side: str = None):
        self.direction = direction
        self.side = side

    @classmethod
    def from_string(cls, direction: str, side: str = None):
        return cls(AnatomicalLandmark.from_string(direction), side=side)

    def compute_default_vector(self) -> np.ndarray:
        return get_constant(self.direction, self.side)

    @property
    def landmarks(self) -> tuple:
        return ()


class StartEndVector(VectorBase):
    """This class represents a vector that is defined by two anatomical landmarks start and end"""

    def __init__(self, start: AnatomicalLandmark, end: AnatomicalLandmark, side: str = None):
        self.start = start
        self.end = end
        self.side = side

    @classmethod
    def from_strings(cls, start: str, end: str, arm_side: str = None):
        return cls(AnatomicalLandmark.from_string(start), AnatomicalLandmark.from_string(end), side=arm_side)

    def __str__(self):
        return f"Start: {self.start}, End: {self.end}"

    @property
    def landmarks(self) -> tuple[AnatomicalLandmark, ...]:
        return self.start, self.end

    def compute_default_vector(self) -> np.ndarray:
        vector = get_constant(self.end, self.side) - get_constant(self.start, self.side)
        return vector / np.linalg.norm(vector)


class CrossedVector(VectorBase):
    """This class represents a vector that is defined by the cross product of two vectors"""

    def __init__(self, first_vector: VectorBase, second_vector: VectorBase):
        self.vector1 = first_vector
        self.vector2 = second_vector

    @property
    def side(self) -> str:
        if self.vector1.side != self.vector2.side:
            raise ValueError("The two vectors must have the same side")
        return self.vector1.side

    def __str__(self):
        return f"({self.vector1}) X ({self.vector2})"

    @property
    def landmarks(self) -> tuple[str, ...]:
        return self.vector1.landmarks + self.vector2.landmarks

    def compute_default_vector(self) -> np.ndarray:
        vector = np.cross(self.vector1.compute_default_vector(), self.vector2.compute_default_vector())
        return vector / np.linalg.norm(vector)


class Frame:
    """This class represents a frame of reference defined by three vectors and an origin"""

    def __init__(
        self,
        x_axis: VectorBase,
        y_axis: VectorBase,
        z_axis: VectorBase,
        origin: AnatomicalLandmark,
        segment: Segment,
    ):
        self.origin = origin
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.segment = segment

    @property
    def side(self):
        if self.x_axis.side != self.y_axis.side or self.y_axis.side != self.z_axis.side:
            raise ValueError(
                "All vectors must have the same side" f"Got {self.x_axis.side}, {self.y_axis.side}, {self.z_axis.side}"
            )
        return self.x_axis.side

    @property
    def axes(self) -> tuple[VectorBase, VectorBase, VectorBase]:
        return self.x_axis, self.y_axis, self.z_axis

    @classmethod
    def from_xy(
        cls, x_axis: VectorBase, y_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment, side: str = None
    ):
        return cls(x_axis, y_axis, CrossedVector(x_axis, y_axis), origin, segment)

    @classmethod
    def from_xz(
        cls, x_axis: VectorBase, z_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment, side: str = None
    ):
        return cls(x_axis, CrossedVector(z_axis, x_axis), z_axis, origin, segment)

    @classmethod
    def from_yz(
        cls, y_axis: VectorBase, z_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment, side: str = None
    ):
        return cls(CrossedVector(y_axis, z_axis), y_axis, z_axis, origin, segment)

    @classmethod
    def from_z_crossed_twice_build_x(cls, y_axis: VectorBase, z_axis, origin: AnatomicalLandmark, segment: Segment):
        x_axis = CrossedVector(y_axis, z_axis)
        return cls.from_xz(x_axis, z_axis, origin, segment)

    @classmethod
    def from_z_crossed_twice_build_y(cls, x_axis: VectorBase, z_axis, origin: AnatomicalLandmark, segment: Segment):
        y_axis = CrossedVector(z_axis, x_axis)
        return cls.from_yz(y_axis, z_axis, origin, segment)

    @classmethod
    def from_y_crossed_twice_build_x(
        cls, y_axis: VectorBase, z_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment
    ):
        x_axis = CrossedVector(y_axis, z_axis)
        return cls.from_xy(x_axis, y_axis, origin, segment)

    @classmethod
    def from_y_crossed_twice_build_z(
        cls, x_axis: VectorBase, y_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment
    ):
        z_axis = CrossedVector(x_axis, y_axis)
        return cls.from_yz(y_axis, z_axis, origin, segment)

    @classmethod
    def from_x_crossed_twice_build_y(
        cls, x_axis: VectorBase, z_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment
    ):
        y_axis = CrossedVector(z_axis, x_axis)
        return cls.from_xy(x_axis, y_axis, origin, segment)

    @classmethod
    def from_x_crossed_twice_build_z(
        cls, x_axis: VectorBase, y_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment
    ):
        z_axis = CrossedVector(x_axis, y_axis)
        return cls.from_xz(x_axis, z_axis, origin, segment)

    @classmethod
    def from_once_crossed(cls, x_axis: str, y_axis: str, z_axis: str, origin: str, segment: Segment, side: str = None):
        if x_axis == "y^z":
            origin = AnatomicalLandmark.from_string(origin)
            return cls.from_yz(parse_axis(y_axis, arm_side=side), parse_axis(z_axis, arm_side=side), origin, segment)
        if y_axis == "z^x":
            origin = AnatomicalLandmark.from_string(origin)
            return cls.from_xz(parse_axis(x_axis, arm_side=side), parse_axis(z_axis, arm_side=side), origin, segment)
        if z_axis == "x^y":
            origin = AnatomicalLandmark.from_string(origin)
            return cls.from_xy(parse_axis(x_axis, arm_side=side), parse_axis(y_axis, arm_side=side), origin, segment)

        raise ValueError(
            f"Invalid axis combination. Expected one of 'x^y', 'y^z', 'z^x' but got {x_axis}, {y_axis}, {z_axis}"
        )

    @classmethod
    def from_twice_crossed(cls, x_axis: str, y_axis: str, z_axis: str, origin: str, segment: Segment, side: str = None):
        is_x_axis_crossed_twice = "x^" in z_axis and "^x" in y_axis
        is_y_axis_crossed_twice = "y^" in x_axis and "^y" in z_axis or "^y" in x_axis and "^y" in z_axis
        is_z_axis_crossed_twice = "z^" in y_axis and "^z" in x_axis or "z^" in y_axis and "z^" in x_axis

        if is_x_axis_crossed_twice:
            if y_axis == "z^x" and "x^" in z_axis and "^x" in y_axis:
                return cls.from_x_crossed_twice_build_z(
                    x_axis=parse_axis(x_axis, arm_side=side),
                    y_axis=parse_axis(z_axis, cross_product_side="second", arm_side=side),
                    origin=AnatomicalLandmark.from_string(origin),
                    segment=segment,
                )
            else:
                return cls.from_x_crossed_twice_build_y(
                    x_axis=parse_axis(x_axis, arm_side=side),
                    z_axis=parse_axis(y_axis, cross_product_side="first", arm_side=side),
                    origin=AnatomicalLandmark.from_string(origin),
                    segment=segment,
                )

        if is_y_axis_crossed_twice:
            if z_axis == "x^y" and "y^" in x_axis and "^y" in z_axis:
                return cls.from_y_crossed_twice_build_x(
                    y_axis=parse_axis(y_axis, arm_side=side),
                    z_axis=parse_axis(x_axis, cross_product_side="second", arm_side=side),
                    origin=AnatomicalLandmark.from_string(origin),
                    segment=segment,
                )
            if z_axis == "x^y" and "^y" in x_axis and "^y" in z_axis:
                return cls.from_y_crossed_twice_build_x(
                    y_axis=parse_axis(y_axis, arm_side=side),
                    z_axis=parse_axis(x_axis, cross_product_side="first", arm_side=side),
                    origin=AnatomicalLandmark.from_string(origin),
                    segment=segment,
                )
            if x_axis == "y^z" and "y^" in x_axis and "^y" in z_axis:
                return cls.from_y_crossed_twice_build_z(
                    x_axis=parse_axis(z_axis, cross_product_side="first", arm_side=side),
                    y_axis=parse_axis(y_axis, arm_side=side),
                    origin=AnatomicalLandmark.from_string(origin),
                    segment=segment,
                )

        if is_z_axis_crossed_twice:
            if x_axis == "y^z" and ("z^" in y_axis and "^z" in x_axis):
                return cls.from_z_crossed_twice_build_y(
                    x_axis=parse_axis(y_axis, cross_product_side="second", arm_side=side),
                    z_axis=parse_axis(z_axis, arm_side=side),
                    origin=AnatomicalLandmark.from_string(origin),
                    segment=segment,
                )

            if y_axis == "z^x" and ("z^" in y_axis and "^z" in x_axis):
                return cls.from_z_crossed_twice_build_x(
                    y_axis=parse_axis(x_axis, cross_product_side="first", arm_side=side),
                    z_axis=parse_axis(z_axis, arm_side=side),
                    origin=AnatomicalLandmark.from_string(origin),
                    segment=segment,
                )
            if y_axis == "z^x" and ("z^" in y_axis and "z^" in x_axis):
                return cls.from_z_crossed_twice_build_x(
                    y_axis=parse_axis(x_axis, cross_product_side="second", arm_side=side),
                    z_axis=parse_axis(z_axis, arm_side=side),
                    origin=AnatomicalLandmark.from_string(origin),
                    segment=segment,
                )

        raise ValueError(
            "Invalid axis combination. Expected one of 'x^y', 'y^z', 'z^x' but got {x_axis}, {y_axis}, {z_axis}."
        )

    @classmethod
    def from_xyz_string(cls, x_axis: str, y_axis: str, z_axis: str, origin: str, segment: Segment, side: str = None):
        are_all_axes_empty = all(axis is None for axis in [x_axis, y_axis, z_axis])

        if are_all_axes_empty and origin is None:
            raise ValueError(
                "All axes and origin are None. Make sure this is what is expected. "
                f"Got x_axis: {x_axis}, y_axis: {y_axis}, z_axis: {z_axis}, origin: {origin}"
            )

        if are_all_axes_empty:
            # previously for Nishinaka that as only the information on distal origin for joint translation information
            return cls(x_axis=None, y_axis=None, z_axis=None, origin=origin, segment=segment)

        if cls.is_one_axis_crossed_twice(x_axis, y_axis, z_axis):
            return cls.from_twice_crossed(x_axis, y_axis, z_axis, origin, segment, side)
        else:
            return cls.from_once_crossed(x_axis, y_axis, z_axis, origin, segment, side)

    @classmethod
    def from_global_thorax_strings(
        cls, x_axis: str, y_axis: str, z_axis: str, origin: str, segment: Segment, side: str = None
    ):
        return cls(
            x_axis=parse_axis(x_axis, arm_side=side),
            y_axis=parse_axis(y_axis, arm_side=side),
            z_axis=parse_axis(z_axis, arm_side=side),
            origin=AnatomicalLandmark.from_string(origin),
            segment=segment,
        )

    @staticmethod
    def is_one_axis_crossed_twice(x_axis: str, y_axis: str, z_axis: str) -> bool:
        is_x_axis_crossed_twice = "x^" in z_axis and "^x" in y_axis
        is_y_axis_crossed_twice = "y^" in x_axis and "^y" in z_axis or "^y" in x_axis and "^y" in z_axis
        is_z_axis_crossed_twice = "z^" in y_axis and "^z" in x_axis or "z^" in y_axis and "z^" in x_axis

        return is_x_axis_crossed_twice or is_y_axis_crossed_twice or is_z_axis_crossed_twice

    @property
    def landmarks(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.x_axis.landmarks + self.y_axis.landmarks + self.z_axis.landmarks).keys())

    @property
    def only_translation(self) -> bool:
        return all([axis is None for axis in self.axes])

    @property
    def is_isb(self) -> bool:
        return (
            self.has_isb_landmarks
            and self.is_x_axis_postero_anterior
            and self.is_y_axis_supero_inferior
            and self.is_z_axis_medio_lateral
            and self.is_origin_isb
        )

    @property
    def is_isb_oriented(self) -> bool:
        return self.is_x_axis_postero_anterior and self.is_y_axis_supero_inferior and self.is_z_axis_medio_lateral

    @property
    def has_isb_landmarks(self) -> bool:
        segment_to_landmark = {
            Segment.SCAPULA: AnatomicalLandmark.Scapula.isb(),
            Segment.THORAX: AnatomicalLandmark.Thorax.isb(),
            Segment.CLAVICLE: AnatomicalLandmark.Clavicle.isb(),
            Segment.HUMERUS: AnatomicalLandmark.Humerus.isb(),
        }

        return set(self.landmarks) == set(segment_to_landmark.get(self.segment))

    @property
    def expected_isb_landmarks(self) -> tuple[str, ...]:
        segment_to_landmark = {
            Segment.SCAPULA: AnatomicalLandmark.Scapula.isb(),
            Segment.THORAX: AnatomicalLandmark.Thorax.isb(),
            Segment.CLAVICLE: AnatomicalLandmark.Clavicle.isb(),
            Segment.HUMERUS: AnatomicalLandmark.Humerus.isb(),
        }
        return segment_to_landmark.get(self.segment)

    @property
    def is_origin_isb(self) -> bool:
        segment_to_origin_isb = {
            Segment.SCAPULA: AnatomicalLandmark.Scapula.origin_isb,
            Segment.THORAX: AnatomicalLandmark.Thorax.origin_isb,
            Segment.CLAVICLE: AnatomicalLandmark.Clavicle.origin_isb,
            Segment.HUMERUS: AnatomicalLandmark.Humerus.origin_isb,
        }

        return self.origin == segment_to_origin_isb.get(self.segment)()

    @property
    def is_x_axis_postero_anterior(self) -> bool:
        return self.x_axis.principal_direction() == CartesianAxis.plusX

    @property
    def is_y_axis_supero_inferior(self) -> bool:
        return self.y_axis.principal_direction() == CartesianAxis.plusY

    @property
    def is_z_axis_medio_lateral(self) -> bool:
        return self.z_axis.principal_direction() == CartesianAxis.plusZ

    @property
    def postero_anterior_axis(self) -> VectorBase:
        for axis in self.axes:
            if axis.biomech_direction() in (
                BiomechDirection.PlusPosteroAnterior,
                BiomechDirection.MinusPosteroAnterior,
            ):
                return axis

    @property
    def infero_superior_axis(self) -> VectorBase:
        for axis in self.axes:
            if axis.biomech_direction() in (BiomechDirection.PlusInferoSuperior, BiomechDirection.MinusInferoSuperior):
                return axis

    @property
    def medio_lateral_axis(self) -> VectorBase:
        for axis in self.axes:
            if axis.biomech_direction() in (BiomechDirection.PlusMedioLateral, BiomechDirection.MinusMedioLateral):
                return axis

    @property
    def get_default_rotation_matrix(self) -> np.ndarray:
        """Returns the default rotation matrix of the frame,
        R_isb_from_local = [ x_local_in_isb, y_local_in_isb, z_local_in_isb]"""
        return np.hstack(
            (
                self.x_axis.compute_default_vector()[:, np.newaxis],
                self.y_axis.compute_default_vector()[:, np.newaxis],
                self.z_axis.compute_default_vector()[:, np.newaxis],
            )
        )

    @property
    def postero_anterior_local_value(self) -> np.ndarray:
        return self.get_default_rotation_matrix[0, :]

    @property
    def infero_superior_local_value(self) -> np.ndarray:
        return self.get_default_rotation_matrix[1, :]

    @property
    def medio_lateral_local_value(self) -> np.ndarray:
        return self.get_default_rotation_matrix[2, :]

    @property
    def medio_lateral_local_axis(self) -> CartesianAxis:
        return CartesianAxis.principal_axis(self.medio_lateral_local_value)

    @property
    def infero_superior_local_axis(self) -> CartesianAxis:
        return CartesianAxis.principal_axis(self.infero_superior_local_value)

    @property
    def postero_anterior_local_axis(self) -> CartesianAxis:
        return CartesianAxis.principal_axis(self.postero_anterior_local_value)

    @property
    def is_left_side(self) -> bool:
        return self.side == "left"

    @property
    def is_direct(self) -> bool:
        return self.get_default_rotation_matrix == 1.0

    def __print__(self):
        return f"Frame: {self.x_axis}, {self.y_axis}, {self.z_axis}, {self.origin}, {self.segment}"


def parse_axis(input_str, cross_product_side="all", arm_side=None) -> VectorBase:
    """
    This function parses the input string of shape "vec(XXX>YYY)", "vec(XXX>YYY)^vec(XXX>YYY)", "vec(XXX>YYY)^vec(XXX>YYY)^vec(XXX>YYY)"
    and returns a Vector or CrossedVector object

    Parameters
    ----------
    input_str: str
        The string to parse
    cross_product_side: str
        The side of the cross product to parse. Can be "first", "second" or "all",
        XXX is the first vector and YYY is the second vector
    arm_side: str
        The side of the arm. Can be "right" arm or "left" arl
    """

    times_crossed = input_str.count("^")

    if times_crossed == 1:
        return parse_crossed_vector(input_str, cross_product_side, arm_side)

    if times_crossed > 1:
        vectors = input_str.split("^")
        multiple_crossed_vector = parse_crossed_vector(f"{vectors[0]}^{vectors[1]}", cross_product_side, arm_side)
        for vector in vectors[2:]:
            multiple_crossed_vector = CrossedVector(multiple_crossed_vector, parse_vector(vector, arm_side))
        return multiple_crossed_vector

    if (cross_product_side == "first" or cross_product_side == "second") and times_crossed == 0:
        raise ValueError(
            f"Invalid input: Expected a crossed vector but got {input_str}"
            f"Set side to 'all' to parse the whole vector"
        )

    return parse_vector(input_str, arm_side)


def parse_crossed_vector(input: str, cross_product_side: str, arm_side: str) -> CrossedVector | StartEndVector:
    first_vector, second_vector = input.split("^")

    if cross_product_side == "all":
        vector1 = parse_vector(first_vector, arm_side)
        vector2 = parse_vector(second_vector, arm_side)
        return CrossedVector(vector1, vector2)
    elif cross_product_side == "first":
        return parse_vector(first_vector, arm_side)
    elif cross_product_side == "second":
        return parse_vector(second_vector, arm_side)
    else:
        raise ValueError(f"Invalid side: Expected 'first', 'second' or 'all' but got {arm_side}")


def parse_vector(input: str, arm_side: str) -> StartEndVector | SoloVector:
    """
    This function parses the input string of shape "vec(XXX>YYY)" and returns a Vector object
    """
    has_parenthesis = input.startswith("(") and input.endswith(")")

    if "vec" not in input and not has_parenthesis:
        return SoloVector(AnatomicalLandmark.from_string(input), side=arm_side)
    if has_parenthesis:
        return SoloVector(AnatomicalLandmark.from_string(input[1:-1]), side=arm_side)

    start, end = parse_start_end_vector(input)
    return StartEndVector.from_strings(start=start, end=end, arm_side=arm_side)


def parse_start_end_vector(input: str) -> tuple[str, ...]:
    """
    This function parses the input string of shape "vec(XXX>YYY)" and returns a tuple of the start and end points
    """
    # check the shape of the string
    if not input.startswith("vec(") or not input.endswith(")") or not ">" in input:
        raise ValueError(f"Invalid input format: Expected 'vec(XXX>YYY)' but got {input}")

    input = input[4:-1].split(">")

    return input[0], input[1]
