from typing import Any

import numpy as np

from .enums_biomech import AnatomicalLandmark, AnatomicalVector


class Global:
    INFERO_SUPERIOR = np.array([0.0, 1.0, 0.0])
    SUPERO_INFERIOR = np.array([0, -1, 0])
    MEDIO_LATERAL = np.array([0.0, 0.0, 1.0])
    LATERO_MEDIAL = np.array([0, 0, -1])
    POSTERO_ANTERIOR = np.array([1.0, 0.0, 0.0])
    IMAGING_CENTER = np.array([np.nan, np.nan, np.nan])


class Thorax:
    """
    This class stores thorax landmarks with default values is ISB frame to make sure we can stick to definitions.
    Pin on Inkscape photo to get the coordinate, unit is pixel.
    """

    IJ_origin = np.array([1395, -111, 0.0])
    IJ = np.array([0.0, 0.0, 0.0])
    SC = np.array([0.0, 0.0, 20])  # made up guess
    PX = np.array([1447, -217, 0.0]) - IJ_origin
    T1s = np.array([1319, -79, 0.0]) - IJ_origin
    C7 = np.array([1320, -73, 0.0]) - IJ_origin
    T1 = np.array([1311, -86, 0.0]) - IJ_origin
    T7 = np.array([1300, -188, 0.0]) - IJ_origin
    T8 = np.array([1297, -212, 0.0]) - IJ_origin
    T10 = np.array([1290, -238, 0.0]) - IJ_origin
    MID_C7_IJ = (C7 + IJ) / 2
    MID_T8_PX = (T8 + PX) / 2
    MID_IJ_T1 = (IJ + T1) / 2
    MID_T10_PX = (T10 + PX) / 2

    y_axis = ((C7 + IJ) / 2 - (T8 + PX) / 2) / np.linalg.norm((C7 + IJ) / 2 - (T8 + PX) / 2)

    vec1 = ((T8 + PX) / 2 - C7) / np.linalg.norm((T8 + PX) / 2 - C7)
    vec2 = IJ - C7
    z_axis = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2))
    x_axis = np.cross(y_axis, z_axis)

    R = np.array([x_axis, y_axis, z_axis]).T

    PX = R @ PX
    T1s = R @ T1s
    C7 = R @ C7
    T1 = R @ T1
    T7 = R @ T7
    T8 = R @ T8
    T10 = R @ T10
    SC = R @ SC
    MID_C7_IJ = R @ MID_C7_IJ
    MID_T8_PX = R @ MID_T8_PX
    MID_IJ_T1 = R @ MID_IJ_T1
    MID_T10_PX = R @ MID_T10_PX

    SPINAL_CANAL_AXIS = np.array([np.sin(np.pi / 6), np.cos(np.pi / 6), 0]) / np.linalg.norm(
        [np.sin(np.pi / 6), np.cos(np.pi / 6), 0]
    )  # made up guess


class Scapula:
    """
    This class stores scapula landmarks with default values is ISB frame to make sure we can stick to definitions.
    """

    AA = np.zeros(3)
    AA_origin = np.array([-77.8502, -71.3278, 1466.713])
    AC = np.array([-40.9962, -71.9892, 1472.783]) - AA_origin
    AI = np.array([-101.1992, 33.4581, 1331.328]) - AA_origin
    GC_CONTOURS = [
        np.array([-49.7961, -62.2796, 1417.6770]) - AA_origin,
        np.array([-33.6853, -57.2951, 1427.9179]) - AA_origin,
        np.array([-60.2887, -59.6469, 1440.5839]) - AA_origin,
        np.array([-48.8372, -60.1961, 1451.6960]) - AA_origin,
        np.array([-62.8106, -60.2237, 1429.8430]) - AA_origin,
        np.array([-33.2097, -57.3340, 1441.9169]) - AA_origin,
    ]
    IE = np.array([-61.4648, -61.3895, 1424.809]) - AA_origin  # Inferior edge of the glenoid
    SE = np.array([-33.9628, -60.4101, 1452.253]) - AA_origin  # Superior edge of the glenoid
    TS = np.array([-71.0315, 43.0474, 1441.641]) - AA_origin

    z_axis = (TS - AA) / np.linalg.norm(TS - AA)
    temp = (TS - AI) / np.linalg.norm(TS - AI)
    x_axis = np.cross(z_axis, temp) / np.linalg.norm(np.cross(z_axis, temp))
    y_axis = np.cross(z_axis, x_axis)

    R = np.array([x_axis, y_axis, z_axis]).T

    manual_offset = np.array([-70, 0.0, 100])  # made up to approximate the global position
    manual_scaling = np.array([0.5, 0.5, 0.5])  # made up to approximate the global position

    AA = (R @ AA) * manual_scaling + manual_offset
    AC = (R @ AC) * manual_scaling + manual_offset
    AI = (R @ AI) * manual_scaling + manual_offset
    for i, gc in enumerate(GC_CONTOURS):
        GC_CONTOURS[i] = (R @ gc) * manual_scaling + manual_offset
    IE = (R @ IE) * manual_scaling + manual_offset + np.array([4, 0, 0])  # make sure IE to SE is mainly inferosuperior
    SE = (R @ SE) * manual_scaling + manual_offset
    TS = (R @ TS) * manual_scaling + manual_offset

    # making sure they are orthogonal
    POSTEROANTERIOR_GLENOID_AXIS = (GC_CONTOURS[1] - GC_CONTOURS[2]) / np.linalg.norm(GC_CONTOURS[1] - GC_CONTOURS[2])
    INFEROSUPERIOR_GLENOID_AXIS_TEMP = (SE - IE) / np.linalg.norm(SE - IE)

    vec = np.cross(POSTEROANTERIOR_GLENOID_AXIS, INFEROSUPERIOR_GLENOID_AXIS_TEMP)
    MEDIOLATERAL_GLENOID_NORMAL = vec / np.linalg.norm(vec)
    LATEROMEDIAL_GLENOID_NORMAL = -MEDIOLATERAL_GLENOID_NORMAL

    vec = np.cross(MEDIOLATERAL_GLENOID_NORMAL, POSTEROANTERIOR_GLENOID_AXIS)
    INFEROSUPERIOR_GLENOID_AXIS = vec / np.linalg.norm(vec)

    ANTEROPOSTERIOR_GLENOID_AXIS = -POSTEROANTERIOR_GLENOID_AXIS


class Clavicle:
    """
    This class stores clavicle landmarks with default values is ISB frame to make sure we can stick to definitions.
    """

    SC_to_AC = Scapula.AC - Thorax.SC
    vec = np.cross(Global.INFERO_SUPERIOR, SC_to_AC)
    POSTEROANTERIOR_AXIS = vec / np.linalg.norm(vec)
    MEDIOLATERAL_AXIS = SC_to_AC / np.linalg.norm(SC_to_AC)
    STERNOCLAVICULAR_SURFACE_CENTROID = Thorax.SC - np.array(
        [5, 0, 0]
    )  # made up guess behind SC along posteroanterior axis


class Humerus:

    EL = np.array([-21, -104, 130])  # made up guess
    EM = np.array([-20, -105, 110])  # made up guess
    GH = np.array([-50, -15, 105])  # made up guess
    MID_EPICONDYLES = (EL + EM) / 2


def get_constant(landmark: Any, side: str) -> np.ndarray:
    the_constant = None

    if isinstance(landmark, AnatomicalVector.Global):
        landmark_mapping = {
            AnatomicalLandmark.Global.IMAGING_ORIGIN: Global.IMAGING_CENTER,
            AnatomicalVector.Global.INFEROSUPERIOR: Global.INFERO_SUPERIOR,
            AnatomicalVector.Global.SUPEROINFERIOR: Global.SUPERO_INFERIOR,
            AnatomicalVector.Global.MEDIOLATERAL: Global.MEDIO_LATERAL,
            AnatomicalVector.Global.LATEROMEDIAL: Global.LATERO_MEDIAL,
            AnatomicalVector.Global.POSTEROANTERIOR: Global.POSTERO_ANTERIOR,
        }
        the_constant = landmark_mapping.get(landmark).copy()

    if isinstance(landmark, AnatomicalLandmark.Thorax) or isinstance(landmark, AnatomicalVector.Thorax):
        landmark_mapping = {
            AnatomicalLandmark.Thorax.IJ: Thorax.IJ,
            AnatomicalLandmark.Thorax.PX: Thorax.PX,
            AnatomicalLandmark.Thorax.T1_ANTERIOR_FACE: Thorax.T1s,
            AnatomicalLandmark.Thorax.T1: Thorax.T1,
            AnatomicalLandmark.Thorax.C7: Thorax.C7,
            AnatomicalLandmark.Thorax.T7: Thorax.T7,
            AnatomicalLandmark.Thorax.T8: Thorax.T8,
            AnatomicalLandmark.Thorax.T10: Thorax.T10,
            AnatomicalLandmark.Thorax.MIDPOINT_C7_IJ: Thorax.MID_C7_IJ,
            AnatomicalLandmark.Thorax.MIDPOINT_IJ_T1: Thorax.MID_IJ_T1,
            AnatomicalLandmark.Thorax.MIDPOINT_T8_PX: Thorax.MID_T8_PX,
            AnatomicalLandmark.Thorax.MIDPOINT_T10_PX: Thorax.MID_T10_PX,
            AnatomicalVector.Thorax.SPINAL_CANAL_AXIS: Thorax.SPINAL_CANAL_AXIS,
        }
        the_constant = landmark_mapping.get(landmark).copy()

    if isinstance(landmark, AnatomicalLandmark.Scapula) or isinstance(landmark, AnatomicalVector.Scapula):
        landmark_mapping = {
            AnatomicalLandmark.Scapula.ANGULAR_ACROMIALIS: Scapula.AA,
            AnatomicalLandmark.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER: Scapula.AC,
            AnatomicalLandmark.Scapula.ANGULUS_INFERIOR: Scapula.AI,
            AnatomicalVector.Scapula.POSTEROANTERIOR_GLENOID_AXIS: Scapula.POSTEROANTERIOR_GLENOID_AXIS,
            AnatomicalVector.Scapula.ANTEROPOSTERIOR_GLENOID_AXIS: Scapula.ANTEROPOSTERIOR_GLENOID_AXIS,
            AnatomicalVector.Scapula.INFEROSUPERIOR_GLENOID_AXIS: Scapula.INFEROSUPERIOR_GLENOID_AXIS,
            AnatomicalVector.Scapula.MEDIOLATERAL_GLENOID_NORMAL: Scapula.MEDIOLATERAL_GLENOID_NORMAL,
            AnatomicalVector.Scapula.LATEROMEDIAL_GLENOID_NORMAL: Scapula.LATEROMEDIAL_GLENOID_NORMAL,
            # AnatomicalLandmark.Scapula.GLENOID_CAVITY_CONTOURS: Scapula.GC_CONTOURS,
            AnatomicalLandmark.Scapula.INFERIOR_EDGE: Scapula.IE,
            AnatomicalLandmark.Scapula.SUPERIOR_EDGE: Scapula.SE,
            AnatomicalLandmark.Scapula.TRIGNONUM_SPINAE: Scapula.TS,
        }

        the_constant = landmark_mapping.get(landmark).copy()

    if isinstance(landmark, AnatomicalLandmark.Clavicle) or isinstance(landmark, AnatomicalVector.Clavicle):
        landmark_mapping = {
            AnatomicalLandmark.Clavicle.STERNOCLAVICULAR_JOINT_CENTER: Thorax.SC,
            AnatomicalLandmark.Clavicle.STERNOCLAVICULAR_SURFACE_CENTROID: Clavicle.STERNOCLAVICULAR_SURFACE_CENTROID,
            AnatomicalVector.Clavicle.POSTEROANTERIOR_AXIS: Clavicle.POSTEROANTERIOR_AXIS,
            AnatomicalVector.Clavicle.MEDIOLATERAL_AXIS: Clavicle.MEDIOLATERAL_AXIS,
        }

        the_constant = landmark_mapping.get(landmark).copy()

    if isinstance(landmark, AnatomicalLandmark.Humerus):
        landmark_mapping = {
            AnatomicalLandmark.Humerus.MIDPOINT_EPICONDYLES: Humerus.MID_EPICONDYLES,
            AnatomicalLandmark.Humerus.LATERAL_EPICONDYLE: Humerus.EL,
            AnatomicalLandmark.Humerus.MEDIAL_EPICONDYLE: Humerus.EM,
            AnatomicalLandmark.Humerus.GLENOHUMERAL_HEAD: Humerus.GH,
        }

        the_constant = landmark_mapping.get(landmark).copy()

    if isinstance(landmark, AnatomicalLandmark):
        if side == "left":
            the_constant[-1] *= -1

    if the_constant is None:
        raise ValueError(f"Landmark {landmark} not found in landmarks")

    if side is not None and side not in ["left", "right"]:
        raise ValueError(f"Side {side} is not a valid side. It should be 'left' or 'right'")

    if side == "left":
        the_constant[-1] *= -1

    return the_constant
