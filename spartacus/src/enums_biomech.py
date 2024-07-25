from enum import Enum

import numpy as np


class CartesianAxis(Enum):
    plusX = ("x", np.array([1, 0, 0]))
    plusY = ("y", np.array([0, 1, 0]))
    plusZ = ("z", np.array([0, 0, 1]))
    minusX = ("-x", np.array([-1, 0, 0]))
    minusY = ("-y", np.array([0, -1, 0]))
    minusZ = ("-z", np.array([0, 0, -1]))

    @classmethod
    def from_array(cls, array: np.ndarray):
        axis_to_enum = {
            (1, 0, 0): cls.plusX,
            (0, 1, 0): cls.plusY,
            (0, 0, 1): cls.plusZ,
            (-1, 0, 0): cls.minusX,
            (0, -1, 0): cls.minusY,
            (0, 0, -1): cls.minusZ,
        }

        the_enum = axis_to_enum.get(tuple(array))
        if the_enum is None:
            raise ValueError(f"{array} is not a valid cartesian axis.")

        return the_enum

    @classmethod
    def principal_axis(cls, array: np.ndarray):
        # max_index = np.argmax(np.abs(array))
        max_index = np.argmax(np.abs(array))
        if array[max_index] > 0:
            return cls.from_array(np.eye(3)[:, max_index])
        else:
            the_array = np.zeros(3)
            the_array[max_index] = -1
            return cls.from_array(the_array)


class BiomechDirection(Enum):
    """Enum for the biomechanical direction"""

    PlusPosteroAnterior = "PlusAntero-Posterior"
    PlusInferoSuperior = "PlusInfero-Superior"
    PlusMedioLateral = "PlusMedio-Lateral"
    MinusPosteroAnterior = "MinusAntero-Posterior"
    MinusInferoSuperior = "MinusInfero-Superior"
    MinusMedioLateral = "MinusMedio-Lateral"

    @classmethod
    def from_string(cls, biomech_direction: str):
        biomech_direction_to_enum = {
            "+mediolateral": cls.PlusMedioLateral,
            "+posteroanterior": cls.PlusPosteroAnterior,
            "+inferosuperior": cls.PlusInferoSuperior,
            "-mediolateral": cls.MinusMedioLateral,
            "-posteroanterior": cls.MinusPosteroAnterior,
            "-inferosuperior": cls.MinusInferoSuperior,
        }

        the_enum = biomech_direction_to_enum.get(biomech_direction)

        if the_enum is None:
            raise ValueError(
                f"{biomech_direction} is not a valid biomech_direction."
                "biomech_direction must be one of the following: "
                "+mediolateral, +anteroposterior, +inferosuperior, "
                "-mediolateral, -anteroposterior, -inferosuperior"
            )

        return the_enum

    @property
    def to_string(self):
        strings = {
            self.PlusPosteroAnterior: "+posteroanterior",
            self.PlusMedioLateral: "+mediolateral",
            self.PlusInferoSuperior: "+inferosuperior",
            self.MinusPosteroAnterior: "-posteroanterior",
            self.MinusMedioLateral: "-mediolateral",
            self.MinusInferoSuperior: "-inferosuperior",
        }

        the_string = strings.get(self)
        if the_string is None:
            raise ValueError(f"{self} is not a valid biomech_direction.")

        return the_string

    @property
    def sign(self):
        sign = {
            self.PlusPosteroAnterior: 1,
            self.PlusMedioLateral: 1,
            self.PlusInferoSuperior: 1,
            self.MinusPosteroAnterior: -1,
            self.MinusMedioLateral: -1,
            self.MinusInferoSuperior: -1,
        }

        return sign[self]

    @classmethod
    def from_direction_global_isb_frame(cls, axis: CartesianAxis):
        """
        Return the biomechanical direction if we are in the global ISB frame
        (X: posteroanterior, Y: inferosuperior, Z: medio-lateral)
        """
        map = {
            CartesianAxis.plusX: cls.PlusPosteroAnterior,
            CartesianAxis.plusY: cls.PlusInferoSuperior,
            CartesianAxis.plusZ: cls.PlusMedioLateral,
            CartesianAxis.minusX: cls.MinusPosteroAnterior,
            CartesianAxis.minusY: cls.MinusInferoSuperior,
            CartesianAxis.minusZ: cls.MinusMedioLateral,
        }
        output = map.get(axis)

        if output is None:
            raise ValueError(f"{axis} is not a valid cartesian axis.")

        return output


class AnatomicalVector:
    """Enum for the biomechanical vectors of the segment, all unit vectors"""

    class Global(Enum):
        INFEROSUPERIOR = "inferosuperior"
        POSTEROANTERIOR = "posteroanterior"
        MEDIOLATERAL = "mediolateral"
        LATEROMEDIAL = "lateromedial"
        SUPEROINFERIOR = "superoinferior"

    class Thorax(Enum):
        SPINAL_CANAL_AXIS = "spinal canal axis"  # pointing infero-superior

    class Scapula(Enum):
        POSTEROANTERIOR_GLENOID_AXIS = "posteroanterior glenoid axis"
        ANTEROPOSTERIOR_GLENOID_AXIS = "anteroposterior glenoid axis"
        INFEROSUPERIOR_GLENOID_AXIS = "inferosuperior glenoid axis"
        MEDIOLATERAL_GLENOID_NORMAL = "mediolateral glenoid normal"
        LATEROMEDIAL_GLENOID_NORMAL = "lateromedial glenoid normal"

    class Clavicle(Enum):
        POSTEROANTERIOR_AXIS = "clavicular posteroanterior axis"
        MEDIOLATERAL_AXIS = "Long axis of the distal part of the clavicle, pointing laterally"

    class Humerus(Enum):
        DIAPHYSIS_INFEROSUPERIOR_AXIS = "diaphysis inferosuperior axis"
        NECK_SHAFT_PLANE_NORMAL = "neck shaft plane normal"


class AnatomicalLandmark:
    """Enum for the biomechanical origins of the segment"""

    class Global(Enum):
        IMAGING_ORIGIN = "imaging centre"

    class Thorax(Enum):

        STERNAL_NOTCH = "SN"
        T7 = "T7"
        T10 = "T10"
        IJ = "IJ"
        T1 = "T1"
        T1_ANTERIOR_FACE = "T1 anterior face"
        C7 = "C7"
        T8 = "T8"
        PX = "PX"  # processus xiphoide
        MIDPOINT_T10_PX = "(T10+PX)/2"
        MIDPOINT_IJ_T1 = "(IJ+T1)/2"
        MIDPOINT_T8_PX = "(T8+PX)/2"
        MIDPOINT_C7_IJ = "(C7+IJ)/2"

        @classmethod
        def isb(cls) -> list:
            return [cls.MIDPOINT_T8_PX, cls.MIDPOINT_C7_IJ, cls.IJ, cls.C7]

        @classmethod
        def origin_isb(cls):
            return cls.IJ

    class Clavicle(Enum):
        STERNOCLAVICULAR_JOINT_CENTER = "SCJC"
        MIDTHIRD = "MTC"
        CUSTOM = "CUSTOM"
        STERNOCLAVICULAR_SURFACE_CENTROID = "CSC"

        @classmethod
        def isb(cls):
            return [
                cls.STERNOCLAVICULAR_JOINT_CENTER,
                AnatomicalLandmark.Thorax.MIDPOINT_T8_PX,
                AnatomicalLandmark.Thorax.MIDPOINT_C7_IJ,
                AnatomicalLandmark.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER,
            ]

        @classmethod
        def origin_isb(cls):
            return cls.STERNOCLAVICULAR_JOINT_CENTER

    class Scapula(Enum):
        ANGULAR_ACROMIALIS = "AA"
        GLENOID_CENTER = "GC"
        ACROMIOCLAVICULAR_JOINT_CENTER = "ACJC"
        TRIGNONUM_SPINAE = "TS"
        ANGULUS_INFERIOR = "IA"
        INFERIOR_EDGE = "IE"
        SUPERIOR_EDGE = "SE"

        @classmethod
        def isb(cls):
            return [cls.ANGULAR_ACROMIALIS, cls.ANGULUS_INFERIOR, cls.TRIGNONUM_SPINAE]

        @classmethod
        def origin_isb(cls):
            return cls.ANGULAR_ACROMIALIS

    class Humerus(Enum):
        GLENOHUMERAL_HEAD = "GH"
        INTERTUBERCULAR_GROOVE = "IG"
        LATERAL_EPICONDYLE = "EL"
        MEDIAL_EPICONDYLE = "EM"
        MIDPOINT_EPICONDYLES = "midpoint epicondyles"  # middle of Medial and Lateral epicondyles

        @classmethod
        def isb(cls):
            return [cls.GLENOHUMERAL_HEAD, cls.LATERAL_EPICONDYLE, cls.MEDIAL_EPICONDYLE, cls.MIDPOINT_EPICONDYLES]

        @classmethod
        def origin_isb(cls):
            return cls.GLENOHUMERAL_HEAD

    class Other(Enum):
        FUNCTIONAL_CENTER = "functional"  # found by score but not meant to represent a real anatomical point

    class Any(Enum):
        NAN = "nan"

    @classmethod
    def from_string(cls, biomech_origin: str):
        if biomech_origin is None:
            return None

        biomech_origin_to_enum = {
            "T7": cls.Thorax.T7,
            "T10": cls.Thorax.T10,
            "IJ": cls.Thorax.IJ,
            "T1": cls.Thorax.T1,
            "T1 anterior face": cls.Thorax.T1_ANTERIOR_FACE,  # old
            "T1s": cls.Thorax.T1_ANTERIOR_FACE,
            "PX": cls.Thorax.PX,
            "(T10+PX)/2": cls.Thorax.MIDPOINT_T10_PX,
            "(PX+T10)/2": cls.Thorax.MIDPOINT_T10_PX,
            "(IJ+T1)/2": cls.Thorax.MIDPOINT_IJ_T1,
            "(T1+IJ)/2": cls.Thorax.MIDPOINT_IJ_T1,
            "C7": cls.Thorax.C7,
            "T8": cls.Thorax.T8,
            "(T8+PX)/2": cls.Thorax.MIDPOINT_T8_PX,
            "(PX+T8)/2": cls.Thorax.MIDPOINT_T8_PX,
            "(C7+IJ)/2": cls.Thorax.MIDPOINT_C7_IJ,
            "(IJ+C7)/2": cls.Thorax.MIDPOINT_C7_IJ,
            "spinal canal axis from T1 to T7": AnatomicalVector.Thorax.SPINAL_CANAL_AXIS,
            "GH": cls.Humerus.GLENOHUMERAL_HEAD,
            "IG": cls.Humerus.INTERTUBERCULAR_GROOVE,
            "EL": cls.Humerus.LATERAL_EPICONDYLE,
            "EM": cls.Humerus.MEDIAL_EPICONDYLE,
            "midpoint EM EL": cls.Humerus.MIDPOINT_EPICONDYLES,  # old
            "(EM+EL)/2": cls.Humerus.MIDPOINT_EPICONDYLES,
            "(EL+EM)/2": cls.Humerus.MIDPOINT_EPICONDYLES,
            "diaphysis inferosuperior axis": AnatomicalVector.Humerus.DIAPHYSIS_INFEROSUPERIOR_AXIS,
            "neck shaft plane normal": AnatomicalVector.Humerus.NECK_SHAFT_PLANE_NORMAL,
            "SC": cls.Clavicle.STERNOCLAVICULAR_JOINT_CENTER,  # most ventral point according to ISB
            "CSC": cls.Clavicle.STERNOCLAVICULAR_SURFACE_CENTROID,  # from Moissenet et al. , supposedly behind SC
            "CM": cls.Clavicle.MIDTHIRD,
            "point of intersection between the mesh model and the Zc axis": cls.Clavicle.CUSTOM,
            "AC": cls.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER,
            "AA": cls.Scapula.ANGULAR_ACROMIALIS,
            "IA": cls.Scapula.ANGULUS_INFERIOR,
            "AI": cls.Scapula.ANGULUS_INFERIOR,  # old
            "glenoid center": cls.Scapula.GLENOID_CENTER,  # old
            "GC": cls.Scapula.GLENOID_CENTER,
            "IE": cls.Scapula.INFERIOR_EDGE,
            "SE": cls.Scapula.SUPERIOR_EDGE,
            "posteroanterior glenoid axis": AnatomicalVector.Scapula.POSTEROANTERIOR_GLENOID_AXIS,
            "anteroposterior glenoid axis": AnatomicalVector.Scapula.ANTEROPOSTERIOR_GLENOID_AXIS,
            "inferosuperior glenoid axis": AnatomicalVector.Scapula.INFEROSUPERIOR_GLENOID_AXIS,
            "mediolateral glenoid normal": AnatomicalVector.Scapula.MEDIOLATERAL_GLENOID_NORMAL,
            "lateromedial glenoid normal": AnatomicalVector.Scapula.LATEROMEDIAL_GLENOID_NORMAL,
            "clavicular posteroanterior axis": AnatomicalVector.Clavicle.POSTEROANTERIOR_AXIS,
            "Long axis of the distal part of the clavicle, pointing laterally": AnatomicalVector.Clavicle.MEDIOLATERAL_AXIS,
            "TS": cls.Scapula.TRIGNONUM_SPINAE,
            "clavicle origin": cls.Clavicle.CUSTOM,
            "functional": cls.Other.FUNCTIONAL_CENTER,
            "imaging inferosuperior axis": AnatomicalVector.Thorax.SPINAL_CANAL_AXIS,
            "imaging superoinferior axis": AnatomicalVector.Global.SUPEROINFERIOR,
            "imaging lateromedial axis": AnatomicalVector.Global.LATEROMEDIAL,
            "imaging mediolateral axis": AnatomicalVector.Global.MEDIOLATERAL,
            "imaging posteroanterior axis": AnatomicalVector.Global.POSTEROANTERIOR,
            "imaging centre": AnatomicalLandmark.Global.IMAGING_ORIGIN,
        }

        the_enum = biomech_origin_to_enum.get(biomech_origin)
        if the_enum is None:
            raise ValueError(
                f"'{biomech_origin}' is not a valid Anatomical landmark."
                "biomech_origin must be one of the following: "
                "joint, parent, child"
            )

        return the_enum


class JointType(Enum):
    """Enum for the joint"""

    GLENO_HUMERAL = "GH"
    SCAPULO_THORACIC = "ST"
    ACROMIO_CLAVICULAR = "AC"
    STERNO_CLAVICULAR = "SC"
    THORACO_HUMERAL = "TH"

    @classmethod
    def from_string(cls, joint: str):
        dico = {
            "glenohumeral": cls.GLENO_HUMERAL,
            "scapulothoracic": cls.SCAPULO_THORACIC,
            "acromioclavicular": cls.ACROMIO_CLAVICULAR,
            "sternoclavicular": cls.STERNO_CLAVICULAR,
            "thoracohumeral": cls.THORACO_HUMERAL,
        }

        the_enum = dico.get(joint)
        if the_enum is None:
            raise ValueError(f"{joint} is not a valid joint.")

        return the_enum

    @property
    def to_string(self):
        dico = {
            self.GLENO_HUMERAL: "glenohumeral",
            self.SCAPULO_THORACIC: "scapulothoracic",
            self.ACROMIO_CLAVICULAR: "acromioclavicular",
            self.STERNO_CLAVICULAR: "sternoclavicular",
            self.THORACO_HUMERAL: "thoracohumeral",
        }

        the_enum = dico.get(self)
        if the_enum is None:
            raise ValueError(f"{self} is not a valid joint.")

        return the_enum

    @property
    def child(self):
        dico = {
            self.STERNO_CLAVICULAR: Segment.CLAVICLE,
            self.ACROMIO_CLAVICULAR: Segment.SCAPULA,
            self.SCAPULO_THORACIC: Segment.SCAPULA,
            self.GLENO_HUMERAL: Segment.HUMERUS,
        }

        the_enum = dico.get(self)
        if the_enum is None:
            raise ValueError(f"{self} is not a valid joint.")
        return the_enum

    @property
    def parent(self):
        dico = {
            self.STERNO_CLAVICULAR: Segment.THORAX,
            self.ACROMIO_CLAVICULAR: Segment.CLAVICLE,
            self.SCAPULO_THORACIC: Segment.THORAX,
            self.GLENO_HUMERAL: Segment.SCAPULA,
        }

        the_enum = dico.get(self)
        if the_enum is None:
            raise ValueError(f"{self} is not a valid joint.")

        return the_enum


class EulerSequence(Enum):
    XYX = "xyx"
    XZX = "xzx"
    XYZ = "xyz"
    XZY = "xzy"
    YXY = "yxy"
    YZX = "yzx"
    YXZ = "yxz"
    YZY = "yzy"
    ZXZ = "zxz"
    ZXY = "zxy"
    ZYZ = "zyz"
    ZYX = "zyx"

    @classmethod
    def isb_from_joint_type(cls, joint_type: JointType):
        joint_type_to_euler_sequence = {
            JointType.GLENO_HUMERAL: cls.YXY,
            JointType.SCAPULO_THORACIC: cls.YXZ,
            JointType.ACROMIO_CLAVICULAR: cls.YXZ,
            JointType.STERNO_CLAVICULAR: cls.YXZ,
            JointType.THORACO_HUMERAL: cls.YXY,
        }

        the_enum = joint_type_to_euler_sequence.get(joint_type)
        if the_enum is None:
            raise ValueError("JointType not recognized")

        return the_enum

    @classmethod
    def from_string(cls, sequence: str):
        if sequence is None:
            return None

        sequence_name_to_enum = {
            "xy'x''": cls.XYX,
            "xz'x''": cls.XZX,
            "xy'z''": cls.XYZ,
            "xz'y''": cls.XZY,
            "yx'y''": cls.YXY,
            "yz'x''": cls.YZX,
            "yx'z''": cls.YXZ,
            "yz'y''": cls.YZY,
            "zx'z''": cls.ZXZ,
            "zx'y''": cls.ZXY,
            "zy'z''": cls.ZYZ,
            "zy'x''": cls.ZYX,
        }

        the_enum = sequence_name_to_enum.get(sequence)
        if the_enum is None:
            raise ValueError(f"{sequence} is not a valid euler sequence.")

        return the_enum

    @property
    def to_string(self) -> str:
        seq = self.value
        return f"{seq[0]}{seq[1]}'{seq[2]}''"


class FrameType:

    PARENT = "parent"
    CHILD = "child"
    JCS = "joint coordinate system"

    @classmethod
    def from_string(cls, frame_type: str):
        frame_type_name_to_enum = {
            "parent": cls.PARENT,
            "child": cls.CHILD,
            "jcs": cls.JCS,
        }

        the_enum = frame_type_name_to_enum.get(frame_type)
        if the_enum is None:
            raise ValueError(f"{frame_type} is not a valid frame type.")

        return the_enum

    # class Local(Enum):
    #     """Enum for the local frame"""
    #
    #     THORAX = "thorax"
    #     HUMERUS = "humerus"
    #     SCAPULA = "scapula"
    #     CLAVICLE = "clavicle"
    #
    # class NonOrthogonal(Enum):
    #     """Enum for the non-orthogonal frame"""
    #
    #     JOINT_STERNOCLAVICULAR = "SC"
    #     JOINT_ACROMIOCLAVICULAR = "AC"
    #     JOINT_GLENOHUMERAL = "GH"
    #     JOINT_SCAPULOTHORACIC = "ST"

    # @classmethod
    # def from_string(cls, frame: str, joint: str):
    #     segment_name_to_enum = {
    #         "thorax": cls.Local.THORAX,
    #         "humerus": cls.Local.HUMERUS,
    #         "scapula": cls.Local.SCAPULA,
    #         "clavicle": cls.Local.CLAVICLE,
    #     }
    #
    #     frame_to_enum = {
    #         ("jcs", "glenohumeral"): cls.NonOrthogonal.JOINT_GLENOHUMERAL,
    #         ("jcs", "scapulothoracic"): cls.NonOrthogonal.JOINT_SCAPULOTHORACIC,
    #         ("jcs", "acromioclavicular"): cls.NonOrthogonal.JOINT_ACROMIOCLAVICULAR,
    #         ("jcs", "sternoclavicular"): cls.NonOrthogonal.JOINT_STERNOCLAVICULAR,
    #     }
    #
    #     the_enum = segment_name_to_enum.get(frame)
    #
    #     if the_enum is None:
    #         the_enum = frame_to_enum.get((frame, joint))
    #
    #     if the_enum is None:
    #         raise ValueError(f"{frame} is not a valid frame.")
    #
    #     return the_enum


class Segment(Enum):
    """Enum for the segment"""

    THORAX = "thorax"
    HUMERUS = "humerus"
    SCAPULA = "scapula"
    CLAVICLE = "clavicle"

    @classmethod
    def from_string(cls, segment: str):
        segment_name_to_enum = {
            "thorax": cls.THORAX,
            "humerus": cls.HUMERUS,
            "scapula": cls.SCAPULA,
            "clavicle": cls.CLAVICLE,
        }

        the_enum = segment_name_to_enum.get(segment)
        if the_enum is None:
            raise ValueError(f"{segment} is not a valid segment.")

        return the_enum

    @property
    def to_string(self):
        return self.value


class Correction(Enum):
    """Enum for the segment coordinate system corrections"""

    # orientation of axis are not orientated as ISB X: anterior, Y: superior, Z: lateral
    TO_ISB_ROTATION = "to_isb"
    TO_ISB_LIKE_ROTATION = "to_isb_like"  # But despite this reorientation, the axis won't be exactly the same as ISB

    SCAPULA_KOLZ_AC_TO_PA_ROTATION = "kolz_AC_to_PA"  # from acromion center of rotation to acromion posterior aspect
    SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION = (
        "glenoid_to_isb_cs"  # from glenoid center of rotation to acromion posterior aspect
    )
    HUMERUS_SULKAR_ROTATION = "Sulkar et al. 2021"  # todo: idk what it is
    SCAPULA_LAGACE_DISPLACEMENT = "Lagace 2012"  # todo: idk what it is

    @classmethod
    def from_string(cls, correction: str):
        correction_name_to_enum = {
            "to_isb": cls.TO_ISB_ROTATION,
            "to_isb_like": cls.TO_ISB_LIKE_ROTATION,
            "kolz_AC_to_PA": cls.SCAPULA_KOLZ_AC_TO_PA_ROTATION,
            "kolz_GC_to_PA": cls.SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION,
            "glenoid_to_isb_cs": cls.SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION,
            "Sulkar et al. 2021": cls.HUMERUS_SULKAR_ROTATION,
            "Lagace 2012": cls.SCAPULA_LAGACE_DISPLACEMENT,
        }

        the_enum = correction_name_to_enum.get(correction)
        if the_enum is None:
            raise ValueError(f"{correction} is not a valid correction method.")

        return the_enum
