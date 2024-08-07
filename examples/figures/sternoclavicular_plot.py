from spartacus import import_data, DataFrameInterface, DataPlanchePlotting
from datetime import datetime


def main(mvt):
    export = False
    df = import_data(correction=True)
    sub_df = df[df["humeral_motion"] == mvt]
    dfi = DataFrameInterface(sub_df).rotational_interface()
    plt = DataPlanchePlotting(dfi, restrict_to_joints=["sternoclavicular"])
    plt.plot()
    plt.update_style()
    plt.show()

    if export:
        plt.fig.write_image(f"../../plots/{mvt}_ST.png")
        plt.fig.write_image(f"../../plots/{mvt}_ST.pdf")
        plt.fig.write_html(f"../../plots/{mvt}_ST.html", include_mathjax="cdn")


if __name__ == "__main__":
    main("frontal plane elevation")
    main("scapular plane elevation")
    main("sagittal plane elevation")
