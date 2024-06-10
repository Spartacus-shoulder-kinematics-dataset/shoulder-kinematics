import numpy as np

from .enums_biomech import AnatomicalLandmark


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
    MID_C7_IJ = (C7 + IJ) / 2
    MID_T8_PX = (T8 + PX) / 2

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
    SC = R @ SC
    MID_C7_IJ = R @ MID_C7_IJ
    MID_T8_PX = R @ MID_T8_PX


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
    IE = (R @ IE) * manual_scaling + manual_offset
    SE = (R @ SE) * manual_scaling + manual_offset
    TS = (R @ TS) * manual_scaling + manual_offset


class Humerus:

    EL = np.array([-21, -104, 130])  # made up guess
    EM = np.array([-20, -105, 110])  # made up guess
    GH = np.array([-50, -15, 105])  # made up guess
    MID_EPICONDYLES = (EL + EM) / 2


def get_constant(landmark: AnatomicalLandmark) -> np.ndarray:
    the_constant = None

    if isinstance(landmark, AnatomicalLandmark.Thorax):
        landmark_mapping = {
            AnatomicalLandmark.Thorax.IJ: Thorax.IJ,
            AnatomicalLandmark.Thorax.PX: Thorax.PX,
            AnatomicalLandmark.Thorax.T1s: Thorax.T1s,
            AnatomicalLandmark.Thorax.C7: Thorax.C7,
            AnatomicalLandmark.Thorax.T7: Thorax.T7,
            AnatomicalLandmark.Thorax.T8: Thorax.T8,
            AnatomicalLandmark.Thorax.MIDPOINT_C7_IJ: Thorax.MID_C7_IJ,
            AnatomicalLandmark.Thorax.MIDPOINT_T8_PX: Thorax.MID_T8_PX,
        }

        the_constant = landmark_mapping.get(landmark)

    if isinstance(landmark, AnatomicalLandmark.Scapula):
        landmark_mapping = {
            AnatomicalLandmark.Scapula.ANGULAR_ACROMIALIS: Scapula.AA,
            AnatomicalLandmark.Scapula.ACROMIOCLAVICULAR_JOINT_CENTER: Scapula.AC,
            AnatomicalLandmark.Scapula.ANGULUS_INFERIOR: Scapula.AI,
            # AnatomicalLandmark.Scapula.GLENOID_CAVITY_CONTOURS: Scapula.GC_CONTOURS,
            # AnatomicalLandmark.Scapula.INFERIOR_EDGE: Scapula.IE,
            # AnatomicalLandmark.Scapula.SUPERIOR_EDGE: Scapula.SE,
            AnatomicalLandmark.Scapula.TRIGNONUM_SPINAE: Scapula.TS,
        }

        the_constant = landmark_mapping.get(landmark)

    if isinstance(landmark, AnatomicalLandmark.Clavicle):
        landmark_mapping = {
            AnatomicalLandmark.Clavicle.STERNOCLAVICULAR_JOINT_CENTER: Thorax.SC,
        }

        the_constant = landmark_mapping.get(landmark)

    if isinstance(landmark, AnatomicalLandmark.Humerus):
        landmark_mapping = {
            AnatomicalLandmark.Humerus.MIDPOINT_EPICONDYLES: Humerus.MID_EPICONDYLES,
            AnatomicalLandmark.Humerus.LATERAL_EPICONDYLE: Humerus.EL,
            AnatomicalLandmark.Humerus.MEDIAL_EPICONDYLE: Humerus.EM,
            AnatomicalLandmark.Humerus.GLENOHUMERAL_HEAD: Humerus.GH,
        }

        the_constant = landmark_mapping.get(landmark)

    if the_constant is None:
        raise ValueError(f"Landmark {landmark} not found in Scapula landmarks")

    return the_constant
