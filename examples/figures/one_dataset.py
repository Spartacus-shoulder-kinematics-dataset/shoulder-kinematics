import os

import pandas as pd
from pandas import DataFrame

from spartacus import DataFrameInterface, DataPlanchePlotting, DatasetCSV, Spartacus

df = pd.read_csv(DatasetCSV.CLEAN.value)
df_henninger = df[df["dataset_authors"] == "Henninger et al."]
df_duplicate_henninger = df_henninger.copy()
df_duplicate_henninger_2 = df_henninger.copy()

# df_henninger["folder"] = "#6_Henninger_et_al/6a_PA"
df_duplicate_henninger["folder"] = "#6_Henninger_et_al/6b_AC"
df_duplicate_henninger["dataset_authors"] = "Henninger et al. 6b AC"
df_duplicate_henninger_2["folder"] = "#6_Henninger_et_al/6c_GC"
df_duplicate_henninger_2["dataset_authors"] = "Henninger et al. 6c GC"

df = pd.concat([df_henninger, df_duplicate_henninger, df_duplicate_henninger_2], ignore_index=True, sort=False)

sp = Spartacus(dataframe=df)
sp.set_correction_callbacks_from_segment_joint_validity(print_warnings=True)
sp.import_confident_data()


def plot_mvt(df: DataFrame, dataset: str = ".", suffix: str = "", export: bool = False):
    # global humeral_motions
    # humeral_motions = df["humeral_motion"].unique() if humeral_motions is [] else humeral_motions

    humeral_motions = df["humeral_motion"].unique()

    for mvt in humeral_motions:
        sub_df = df[df["humeral_motion"] == mvt]
        dfi = DataFrameInterface(sub_df)
        plt = DataPlanchePlotting(dfi)
        plt.plot()
        plt.update_style()
        plt.show()
        if export:
            dataset += "/"
            plt.fig.write_image(f"../figures/{dataset}{mvt}{suffix}.png")
            plt.fig.write_image(f"../figures/{dataset}{mvt}{suffix}.pdf")
            plt.fig.write_html(f"../figures/{dataset}{mvt}{suffix}.html", include_mathjax="cdn")


def before_after():
    df_before = sp.confident_data_values
    df_after = sp.corrected_confident_data_values

    df_after_copy = df_after.copy()
    df_after_copy["article"] = df_after_copy["article"].apply(lambda x: x + "_corrected")
    df_both = pd.concat([df_before, df_after_copy], ignore_index=True, sort=False)

    dataset = df_after["article"].unique()[0]
    dataset_corrected = df_after_copy["article"].unique()[0]

    if not os.path.exists(dataset):
        os.mkdir(dataset)

    # sub_df_before = df_before[df_before["article"] == dataset]
    # sub_df_after = df_after[df_after["article"] == dataset]
    condition = (df_both["article"] == dataset) + (df_both["article"] == dataset_corrected)
    sub_df_both = df_both[condition]
    plot_mvt(df_before, dataset, suffix="_before", export=True)
    # plot_mvt(df_after, dataset, suffix="_after")
    # plot_mvt(sub_df_both, dataset, suffix="_both")


if __name__ == "__main__":
    before_after()
