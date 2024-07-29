import os

import pandas as pd
from pandas import DataFrame

from spartacus import import_data, DataFrameInterface, DataPlanchePlotting


def plot_mvt(df: DataFrame, dataset: str = ".", suffix: str = "", export: bool = False):

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
            plt.fig.write_image(f"../plots/{dataset}{mvt}{suffix}.png")
            plt.fig.write_image(f"../plots/{dataset}{mvt}{suffix}.pdf")
            plt.fig.write_html(f"../plots/{dataset}{mvt}{suffix}.html", include_mathjax="cdn")


def main():
    df = import_data(correction=True)
    df = df[df["unit"] == "rad"]
    plot_mvt(df)


def before_after():
    df_before = import_data(correction=False)
    df_after = import_data(correction=True)

    df_after_copy = df_after.copy()
    df_after_copy["article"] = df_after_copy["article"].apply(lambda x: x + "_corrected")
    df_both = pd.concat([df_before, df_after_copy], ignore_index=True, sort=False)

    datasets = df_after["article"].unique()
    datasets_corrected = df_after_copy["article"].unique()
    for dataset, dataset_corrected in zip(datasets, datasets_corrected):

        if not os.path.exists(dataset):
            os.mkdir(dataset)

        sub_df_before = df_before[df_before["article"] == dataset]
        sub_df_after = df_after[df_after["article"] == dataset]
        condition = (df_both["article"] == dataset) + (df_both["article"] == dataset_corrected)
        sub_df_both = df_both[condition]
        # plot_mvt(sub_df_before, dataset, suffix="_before")
        # plot_mvt(sub_df_after, dataset, suffix="_after")
        plot_mvt(sub_df_both, dataset, suffix="_both")


if __name__ == "__main__":
    main()
    # before_after()
