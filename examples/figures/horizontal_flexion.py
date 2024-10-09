from pandas import DataFrame

from spartacus import import_data, DataFrameInterface, DataPlanchePlotting


def plot_mvt(df: DataFrame, dataset: str = ".", suffix: str = "", export: bool = False):

    humeral_motions = ["horizontal flexion"]

    for mvt in humeral_motions:
        sub_df = df[df["humeral_motion"] == mvt]
        dfi = DataFrameInterface(sub_df).rotational_interface()
        plt = DataPlanchePlotting(dfi)
        plt.plot()
        plt.update_style()
        plt.show()
        if export:
            dataset += "/"
            plt.fig.write_image(f"../../plots/{mvt}{suffix}.png")
            plt.fig.write_image(f"../../plots/{mvt}{suffix}.pdf")
            plt.fig.write_html(f"../../plots/{mvt}{suffix}.html", include_mathjax="cdn")


def main():
    df = import_data(correction=True)
    plot_mvt(df, export=True)


if __name__ == "__main__":
    main()
