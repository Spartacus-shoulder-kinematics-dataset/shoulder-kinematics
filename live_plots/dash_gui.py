import dash
from dash import dcc, html
from dash.dependencies import Input, Output

from ..plots.planche_plotting import DataPlanchePlotting
from ..plots.dataframe_interface import DataFrameInterface

from ..quick_load import import_data

df = import_data(correction=True)
df = df[df["unit"] == "rad"]
dfi = DataFrameInterface(df)
plt = DataPlanchePlotting(dfi)
plt.plot()
plt.update_style()
# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Biomechanical Data Visualization"),
        #  Dropdown to select humeral motion
        dcc.Dropdown(
            id="humeral_motion",
            options=list(dfi.df["humeral_motion"].unique()),
            value="scapular plane elevation",
        ),
        # Checklist to select joints
        html.Div(
            [
                html.Label("Select Joints:"),
                dcc.Checklist(
                    id="joint-checklist",
                    options=["glenohumeral", "scapulothoracic", "acromioclavicular", "sternoclavicular"],
                    value=[
                        "glenohumeral",
                        "scapulothoracic",
                        "acromioclavicular",
                        "sternoclavicular",
                    ],  # Default to the first 4 joints
                    inline=True,
                ),
            ],
            style={"margin-top": "20px"},
        ),
        # Graph component
        dcc.Graph(
            id="biomech-plot",
            figure=plt.fig,
        ),
    ]
)


# Callback to update the figure
@app.callback(Output("biomech-plot", "figure"), [Input("humeral_motion", "value"), Input("joint-checklist", "value")])
def update_figure(selected_humeral_motion, selected_joints):

    subdf = df[df["humeral_motion"] == selected_humeral_motion]
    subdf = subdf[subdf["joint"].isin(selected_joints)]

    # Initialize the DataPlanchePlotting object with selected options
    dfi = DataFrameInterface(subdf)
    plt = DataPlanchePlotting(dfi, restrict_to_joints=selected_joints)
    plt.plot()
    plt.update_style()

    return plt.fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
