from spartacus import import_data, DataFrameInterface, DataPlanchePlotting


def main():
    export = True
    mvt = "sagittal plane elevation"
    df = import_data(correction=True)
    sub_df = df[df["humeral_motion"] == mvt]
    sub_df = sub_df[sub_df["article"] != "Henninger et al."]
    # sub_df = sub_df[sub_df["article"] == "Begon et al."]
    # sub_df = sub_df[sub_df["shoulder_id"] == 1]
    dfi = DataFrameInterface(sub_df)
    dfi_rot = dfi.rotational_interface()
    plt = DataPlanchePlotting(dfi_rot, restrict_to_joints=["glenohumeral"])
    plt.plot()
    plt.update_style()
    plt.show()

    if export:
        plt.fig.write_image(f"../../plots/{mvt}_scapulothoracic.png")
        plt.fig.write_image(f"../../plots/{mvt}_scapulothoracic.pdf")
        plt.fig.write_html(f"../../plots/{mvt}_scapulothoracic.html", include_mathjax="cdn")


if __name__ == "__main__":
    main()
