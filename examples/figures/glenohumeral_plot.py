from spartacus import import_data, DataFrameInterface, DataPlanchePlotting


def main(mvt):
    export = False
    df = import_data(correction=True)
    sub_df = df[df["humeral_motion"] == mvt]
    dfi = DataFrameInterface(sub_df)
    dfi_rot = dfi.rotational_interface()
    plt = DataPlanchePlotting(dfi_rot, restrict_to_joints=["glenohumeral"])
    plt.plot()
    plt.update_style()
    plt.show()

    if export:
        plt.fig.write_image(f"../../plots/{mvt}_GH.png")
        plt.fig.write_image(f"../../plots/{mvt}_GH.pdf")
        plt.fig.write_html(f"../../plots/{mvt}_GH.html", include_mathjax="cdn")


if __name__ == "__main__":
    main("frontal plane elevation")
    main("scapular plane elevation")
    main("sagittal plane elevation")
