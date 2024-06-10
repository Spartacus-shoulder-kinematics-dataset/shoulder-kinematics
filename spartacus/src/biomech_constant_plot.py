from plotly import graph_objects as go

from biomech_constant import Scapula, Thorax, Humerus


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


def plot_thorax():
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[Thorax.IJ[0]], y=[Thorax.IJ[1]], z=[Thorax.IJ[2]], mode="markers", name="IJ"))
    fig.add_trace(go.Scatter3d(x=[Thorax.PX[0]], y=[Thorax.PX[1]], z=[Thorax.PX[2]], mode="markers", name="PX"))
    fig.add_trace(go.Scatter3d(x=[Thorax.T1s[0]], y=[Thorax.T1s[1]], z=[Thorax.T1s[2]], mode="markers", name="T1s"))
    fig.add_trace(go.Scatter3d(x=[Thorax.C7[0]], y=[Thorax.C7[1]], z=[Thorax.C7[2]], mode="markers", name="C7"))
    fig.add_trace(go.Scatter3d(x=[Thorax.T1[0]], y=[Thorax.T1[1]], z=[Thorax.T1[2]], mode="markers", name="T1"))
    fig.add_trace(go.Scatter3d(x=[Thorax.T7[0]], y=[Thorax.T7[1]], z=[Thorax.T7[2]], mode="markers", name="T7"))
    fig.add_trace(go.Scatter3d(x=[Thorax.T8[0]], y=[Thorax.T8[1]], z=[Thorax.T8[2]], mode="markers", name="T8"))
    fig.add_trace(go.Scatter3d(x=[Thorax.SC[0]], y=[Thorax.SC[1]], z=[Thorax.SC[2]], mode="markers", name="SC"))
    fig.add_trace(
        go.Scatter3d(
            x=[Thorax.MID_C7_IJ[0]], y=[Thorax.MID_C7_IJ[1]], z=[Thorax.MID_C7_IJ[2]], mode="markers", name="MID_C7_IJ"
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Thorax.MID_T8_PX[0]], y=[Thorax.MID_T8_PX[1]], z=[Thorax.MID_T8_PX[2]], mode="markers", name="MID_T8_PX"
        )
    )
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()


def plot_all():

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[Scapula.AA[0]], y=[Scapula.AA[1]], z=[Scapula.AA[2]], mode="markers", name="AA", marker=dict(color="red")
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Scapula.AC[0]], y=[Scapula.AC[1]], z=[Scapula.AC[2]], mode="markers", name="AC", marker=dict(color="red")
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Scapula.AI[0]], y=[Scapula.AI[1]], z=[Scapula.AI[2]], mode="markers", name="AI", marker=dict(color="red")
        )
    )
    for i, gc in enumerate(Scapula.GC_CONTOURS):
        fig.add_trace(
            go.Scatter3d(x=[gc[0]], y=[gc[1]], z=[gc[2]], mode="markers", name=f"GC {i}", marker=dict(color="red"))
        )
    fig.add_trace(
        go.Scatter3d(
            x=[Scapula.IE[0]], y=[Scapula.IE[1]], z=[Scapula.IE[2]], mode="markers", name="IE", marker=dict(color="red")
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Scapula.SE[0]], y=[Scapula.SE[1]], z=[Scapula.SE[2]], mode="markers", name="SE", marker=dict(color="red")
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Scapula.TS[0]], y=[Scapula.TS[1]], z=[Scapula.TS[2]], mode="markers", name="TS", marker=dict(color="red")
        )
    )

    fig.add_trace(go.Scatter3d(x=[Thorax.IJ[0]], y=[Thorax.IJ[1]], z=[Thorax.IJ[2]], mode="markers", name="IJ"))
    fig.add_trace(go.Scatter3d(x=[Thorax.PX[0]], y=[Thorax.PX[1]], z=[Thorax.PX[2]], mode="markers", name="PX"))
    fig.add_trace(go.Scatter3d(x=[Thorax.T1s[0]], y=[Thorax.T1s[1]], z=[Thorax.T1s[2]], mode="markers", name="T1s"))
    fig.add_trace(go.Scatter3d(x=[Thorax.C7[0]], y=[Thorax.C7[1]], z=[Thorax.C7[2]], mode="markers", name="C7"))
    fig.add_trace(go.Scatter3d(x=[Thorax.T1[0]], y=[Thorax.T1[1]], z=[Thorax.T1[2]], mode="markers", name="T1"))
    fig.add_trace(go.Scatter3d(x=[Thorax.T7[0]], y=[Thorax.T7[1]], z=[Thorax.T7[2]], mode="markers", name="T7"))
    fig.add_trace(go.Scatter3d(x=[Thorax.T8[0]], y=[Thorax.T8[1]], z=[Thorax.T8[2]], mode="markers", name="T8"))
    fig.add_trace(
        go.Scatter3d(
            x=[Thorax.SC[0]], y=[Thorax.SC[1]], z=[Thorax.SC[2]], mode="markers", name="SC", marker=dict(color="blue")
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Thorax.MID_C7_IJ[0]], y=[Thorax.MID_C7_IJ[1]], z=[Thorax.MID_C7_IJ[2]], mode="markers", name="MID_C7_IJ"
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Thorax.MID_T8_PX[0]], y=[Thorax.MID_T8_PX[1]], z=[Thorax.MID_T8_PX[2]], mode="markers", name="MID_T8_PX"
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[Humerus.EL[0]],
            y=[Humerus.EL[1]],
            z=[Humerus.EL[2]],
            mode="markers",
            name="EL",
            marker=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Humerus.EM[0]],
            y=[Humerus.EM[1]],
            z=[Humerus.EM[2]],
            mode="markers",
            name="EM",
            marker=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Humerus.GH[0]],
            y=[Humerus.GH[1]],
            z=[Humerus.GH[2]],
            mode="markers",
            name="GH",
            marker=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[Humerus.MID_EPICONDYLES[0]],
            y=[Humerus.MID_EPICONDYLES[1]],
            z=[Humerus.MID_EPICONDYLES[2]],
            mode="markers",
            name="MID_EPICONDYLES",
            marker=dict(color="green"),
        )
    )

    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()


# plot_scapula()
# plot_thorax()
plot_all()
