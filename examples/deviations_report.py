import pandas as pd
import plotly.express as px

from spartacus import DataFolder, load_subdataset

deviation_cols = [
    "parent_d1",  # float
    "parent_d2",  # float
    "parent_d3",  # float
    "parent_d4",  # float
    "child_d1",  # float
    "child_d2",  # float
    "child_d3",  # float
    "child_d4",  # float
    "d5",  # float
    "d6",  # float
    "d7",  # float
    "total_deviation",  # float
]

deviation_df = pd.DataFrame(columns=["dataset_authors", "joint"] + deviation_cols)

for folder in DataFolder:
    try:
        data = load_subdataset(name=folder)
        df = data.corrected_confident_data_values

        for joint in df["joint"].unique():
            subdf = df[df["joint"] == joint]
            deviation_values = subdf[deviation_cols].values[0, :]

            new_dict = dict()
            new_dict["dataset_authors"] = folder.to_dataset_author()
            new_dict["joint"] = joint
            for i, col in enumerate(deviation_cols):
                new_dict[col] = deviation_values[i]

            deviation_df = pd.concat([deviation_df, pd.DataFrame([new_dict])], ignore_index=True)

    except:
        print("could not make it yet for ", folder)
        continue

for joint in deviation_df["joint"].unique():
    subdf = deviation_df[deviation_df["joint"] == joint]
    fig = px.imshow(
        subdf[deviation_cols].values,
        labels=dict(x=f"Deviations for {joint} joint", y="Authors", color="Value"),
        x=deviation_cols,
        y=subdf["dataset_authors"],
    )
    fig.update_xaxes(side="top")
    fig.show()


deviation_df.to_csv("deviation_report.csv", index=False)
