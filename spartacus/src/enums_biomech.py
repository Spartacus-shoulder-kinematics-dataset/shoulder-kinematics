from enum import Enum
from pathlib import Path

import numpy as np


class CartesianAxis(Enum):
    plusX = ("x", np.array([1, 0, 0]))
    plusY = ("y", np.array([0, 1, 0]))
    plusZ = ("z", np.array([0, 0, 1]))
    minusX = ("-x", np.array([-1, 0, 0]))
    minusY = ("-y", np.array([0, -1, 0]))
    minusZ = ("-z", np.array([0, 0, -1]))


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


class BiomechOrigin:
    """Enum for the biomechanical origins of the segment"""

    class Thorax(Enum):
        STERNAL_NOTCH = "SN"
        T7 = "T7"
        IJ = "IJ"
        T1_ANTERIOR_FACE = "T1 anterior face"
        T1s = "T1s"  # @todo: make sure to understand what is it
        C7 = "C7"
        T8 = "T8"
        PX = "PX"  # processus xiphoide

    class Clavicle(Enum):
        STERNOCLAVICULAR_JOINT_CENTER = "SCJC"
        MIDTHIRD = "MTC"
        CUSTOM = "CUSTOM"
        ACROMIOCLAVICULAR_JOINT_CENTER = "ACJC"

    class Scapula(Enum):
        ANGULAR_ACROMIALIS = "AA"
        GLENOID_CENTER = "GC"
        ACROMIOCLAVICULAR_JOINT_CENTER = "ACJC"
        TRIGNONUM_SPINAE = "TS"
        ANGULUS_INFERIOR = "AI"

    class Humerus(Enum):
        GLENOHUMERAL_HEAD = "GH"
        MIDPOINT_EPICONDYLES = "midpoint epicondyles"  # middle of Medial and Lateral epicondyles

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
            "IJ": cls.Thorax.IJ,
            "T1 anterior face": cls.Thorax.T1_ANTERIOR_FACE,  # old
            "T1s": cls.Thorax.T1_ANTERIOR_FACE,
            "GH": cls.Humerus.GLENOHUMERAL_HEAD,
            "midpoint EM EL": cls.Humerus.MIDPOINT_EPICONDYLES,  # old
            "(EM+EL)/2": cls.Humerus.MIDPOINT_EPICONDYLES,
            "SC": cls.Clavicle.STERNOCLAVICULAR_JOINT_CENTER,
            "CM": cls.Clavicle.MIDTHIRD,
            "point of intersection between the mesh model and the Zc axis": cls.Clavicle.CUSTOM,
            "AC": cls.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER,
            "AA": cls.Scapula.ANGULAR_ACROMIALIS,
            "glenoid center": cls.Scapula.GLENOID_CENTER,  # old
            "GC": cls.Scapula.GLENOID_CENTER,
            "TS": cls.Scapula.TRIGNONUM_SPINAE,
            "clavicle origin": cls.Clavicle.CUSTOM,
            "functional": cls.Other.FUNCTIONAL_CENTER,
        }

        the_enum = biomech_origin_to_enum.get(biomech_origin)
        if the_enum is None:
            raise ValueError(
                f"{biomech_origin} is not a valid biomech_origin."
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
            "xyx": cls.XYX,
            "xzx": cls.XZX,
            "xyz": cls.XYZ,
            "xzy": cls.XZY,
            "yxy": cls.YXY,
            "yzx": cls.YZX,
            "yxz": cls.YXZ,
            "yzy": cls.YZY,
            "zxz": cls.ZXZ,
            "zxy": cls.ZXY,
            "zyz": cls.ZYZ,
            "zyx": cls.ZYX,
        }

        the_enum = sequence_name_to_enum.get(sequence)
        if the_enum is None:
            raise ValueError(f"{sequence} is not a valid euler sequence.")

        return the_enum


class Frame:
    class Local(Enum):
        """Enum for the local frame"""

        THORAX = "thorax"
        HUMERUS = "humerus"
        SCAPULA = "scapula"
        CLAVICLE = "clavicle"

    class NonOrthogonal(Enum):
        """Enum for the non-orthogonal frame"""

        JOINT_STERNOCLAVICULAR = "SC"
        JOINT_ACROMIOCLAVICULAR = "AC"
        JOINT_GLENOHUMERAL = "GH"
        JOINT_SCAPULOTHORACIC = "ST"

    @classmethod
    def from_string(cls, frame: str, joint: str):
        segment_name_to_enum = {
            "thorax": cls.Local.THORAX,
            "humerus": cls.Local.HUMERUS,
            "scapula": cls.Local.SCAPULA,
            "clavicle": cls.Local.CLAVICLE,
        }

        frame_to_enum = {
            ("jcs", "glenohumeral"): cls.NonOrthogonal.JOINT_GLENOHUMERAL,
            ("jcs", "scapulothoracic"): cls.NonOrthogonal.JOINT_SCAPULOTHORACIC,
            ("jcs", "acromioclavicular"): cls.NonOrthogonal.JOINT_ACROMIOCLAVICULAR,
            ("jcs", "sternoclavicular"): cls.NonOrthogonal.JOINT_STERNOCLAVICULAR,
        }

        the_enum = segment_name_to_enum.get(frame)

        if the_enum is None:
            the_enum = frame_to_enum.get((frame, joint))

        if the_enum is None:
            raise ValueError(f"{frame} is not a valid frame.")

        return the_enum


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
