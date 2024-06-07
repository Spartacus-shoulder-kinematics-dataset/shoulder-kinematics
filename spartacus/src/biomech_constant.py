import numpy as np
import plotly.graph_objects as go

from .enums_biomech import AnatomicalLandmark


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

    AC = R @ AC
    AI = R @ AI
    for i, gc in enumerate(GC_CONTOURS):
        GC_CONTOURS[i] = R @ gc
    IE = R @ IE
    SE = R @ SE
    TS = R @ TS


def plot_scapula():
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[Scapula.AA[0]], y=[Scapula.AA[1]], z=[Scapula.AA[2]], mode="markers", name="AA"))
    fig.add_trace(go.Scatter3d(x=[Scapula.AC[0]], y=[Scapula.AC[1]], z=[Scapula.AC[2]], mode="markers", name="AC"))
    fig.add_trace(go.Scatter3d(x=[Scapula.AI[0]], y=[Scapula.AI[1]], z=[Scapula.AI[2]], mode="markers", name="AI"))
    for i, gc in enumerate(Scapula.GC_CONTOURS):
        fig.add_trace(go.Scatter3d(x=[gc[0]], y=[gc[1]], z=[gc[2]], mode="markers", name=f"GC {i}"))
    fig.add_trace(go.Scatter3d(x=[Scapula.IE[0]], y=[Scapula.IE[1]], z=[Scapula.IE[2]], mode="markers", name="IE"))
    fig.add_trace(go.Scatter3d(x=[Scapula.SE[0]], y=[Scapula.SE[1]], z=[Scapula.SE[2]], mode="markers", name="SE"))
    fig.add_trace(go.Scatter3d(x=[Scapula.TS[0]], y=[Scapula.TS[1]], z=[Scapula.TS[2]], mode="markers", name="TS"))
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()


def get_constant(landmark: AnatomicalLandmark) -> np.ndarray:
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

        if the_constant is None:
            raise ValueError(f"Landmark {landmark} not found in Scapula landmarks")

        return landmark_mapping[landmark]


# plot_scapula()
