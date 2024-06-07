from abc import ABC, abstractmethod

import numpy as np

from .biomech_constant import get_constant
from .enums_biomech import AnatomicalLandmark, Segment, CartesianAxis


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


class Vector(VectorBase):
    def __init__(self, start: AnatomicalLandmark, end: AnatomicalLandmark):
        self.start = start
        self.end = end

    @classmethod
    def from_strings(cls, start: str, end: str):
        return cls(AnatomicalLandmark.from_string(start), AnatomicalLandmark.from_string(end))

    def __str__(self):
        return f"Start: {self.start}, End: {self.end}"

    @property
    def landmarks(self) -> tuple[AnatomicalLandmark, ...]:
        return self.start, self.end

    def compute_default_vector(self) -> np.ndarray:
        vector = get_constant(self.end) - get_constant(self.start)
        return vector / np.linalg.norm(vector)


class CrossedVector(VectorBase):
    def __init__(self, first_vector: VectorBase, second_vector: VectorBase):
        self.vector1 = first_vector
        self.vector2 = second_vector

    def __str__(self):
        return f"({self.vector1}) X ({self.vector2})"

    @property
    def landmarks(self) -> tuple[str, ...]:
        return self.vector1.landmarks + self.vector2.landmarks

    def compute_default_vector(self) -> np.ndarray:
        vector = np.cross(self.vector1.compute_default_vector(), self.vector2.compute_default_vector())
        return vector / np.linalg.norm(vector)


class Frame:
    def __init__(
        self, x_axis: VectorBase, y_axis: VectorBase, z_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment
    ):
        self.origin = origin
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.segment = segment

    @classmethod
    def from_xy(cls, x_axis: VectorBase, y_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment):
        return cls(x_axis, y_axis, CrossedVector(x_axis, y_axis), origin, segment)

    @classmethod
    def from_xz(cls, x_axis: VectorBase, z_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment):
        return cls(x_axis, CrossedVector(z_axis, x_axis), z_axis, origin, segment)

    @classmethod
    def from_yz(cls, y_axis: VectorBase, z_axis: VectorBase, origin: AnatomicalLandmark, segment: Segment):
        return cls(CrossedVector(y_axis, z_axis), y_axis, z_axis, origin, segment)

    @classmethod
    def from_xyz_string(cls, x_axis: str, y_axis: str, z_axis: str, origin: str, segment: Segment):
        if x_axis == "y^z":
            origin = AnatomicalLandmark.from_string(origin)
            return cls.from_yz(parse_axis(y_axis), parse_axis(z_axis), origin, segment)
        if y_axis == "z^x":
            origin = AnatomicalLandmark.from_string(origin)
            return cls.from_xz(parse_axis(x_axis), parse_axis(z_axis), origin, segment)
        if z_axis == "x^y":
            origin = AnatomicalLandmark.from_string(origin)
            return cls.from_xy(parse_axis(x_axis), parse_axis(y_axis), origin, segment)

        raise ValueError(
            f"Invalid axis combination. Expected one of 'x^y', 'y^z', 'z^x' but got {x_axis}, {y_axis}, {z_axis}"
        )

    @property
    def landmarks(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.x_axis.landmarks + self.y_axis.landmarks + self.z_axis.landmarks).keys())

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
    def has_isb_landmarks(self) -> bool:
        if self.segment == Segment.SCAPULA:
            return all([landmark in self.landmarks for landmark in AnatomicalLandmark.Scapula.isb()])

    @property
    def is_origin_isb(self) -> bool:
        if self.segment == Segment.SCAPULA:
            return self.origin == AnatomicalLandmark.Scapula.origin_isb()

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
        for axis in [self.x_axis, self.y_axis, self.z_axis]:
            if axis.principal_direction() in (CartesianAxis.plusX, CartesianAxis.minusX):
                return axis

    @property
    def infero_superior_axis(self) -> VectorBase:
        for axis in [self.x_axis, self.y_axis, self.z_axis]:
            if axis.principal_direction() in (CartesianAxis.plusY, CartesianAxis.minusY):
                return axis

    @property
    def medio_lateral_axis(self) -> VectorBase:
        for axis in [self.x_axis, self.y_axis, self.z_axis]:
            if axis.principal_direction() in (CartesianAxis.plusZ, CartesianAxis.minusZ):
                return axis

    @property
    def postero_anterior_local_axis(self) -> CartesianAxis:
        for axis in [self.x_axis, self.y_axis, self.z_axis]:
            if axis.principal_direction() in (CartesianAxis.plusX, CartesianAxis.minusX):
                return axis.principal_direction()

    @property
    def infero_superior_local_axis(self) -> CartesianAxis:
        for axis in [self.x_axis, self.y_axis, self.z_axis]:
            if axis.principal_direction() in (CartesianAxis.plusY, CartesianAxis.minusY):
                return axis.principal_direction()

    @property
    def medio_lateral_local_axis(self) -> CartesianAxis:
        for axis in [self.x_axis, self.y_axis, self.z_axis]:
            if axis.principal_direction() in (CartesianAxis.plusZ, CartesianAxis.minusZ):
                return axis.principal_direction()

    @property
    def is_left_side(self) -> bool:
        pass
        # TODO: Implement this property

    def __print__(self):
        return f"Frame: {self.x_axis}, {self.y_axis}, {self.z_axis}, {self.origin}, {self.segment}"


def parse_axis(input_str) -> VectorBase:
    """
    This function parses the input string of shape "vec(XXX>YYY)" or "vec(XXX>YYY)^vec(XXX>YYY)"
    and returns a Vector or CrossedVector object
    """
    is_crossed = "^" in input_str
    if is_crossed:
        first_vector, second_vector = input_str.split("^")
        vector1 = Vector.from_strings(*parse_vectors(first_vector))
        vector2 = Vector.from_strings(*parse_vectors(second_vector))
        return CrossedVector(vector1, vector2)
    else:
        return Vector.from_strings(*parse_vectors(input_str))


def parse_vectors(input: str) -> tuple[str, ...]:
    """
    This function parses the input string of shape "vec(XXX>YYY)" and returns a tuple of the start and end points
    """
    # check the shape of the string
    if not input.startswith("vec(") or not input.endswith(")") or not ">" in input:
        raise ValueError(f"Invalid input format: Expected 'vec(XXX>YYY)' but got {input}")

    input = input[4:-1].split(">")

    return input[0], input[1]
