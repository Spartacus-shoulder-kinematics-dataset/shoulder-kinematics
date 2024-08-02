from spartacus import import_data, DataFrameInterface, DataPlanchePlotting


def manual_corrections(sub_df):
    # ONLY FOR DISPLAYING PURPOSE, EXPECT THIS TO DISAPPEAR ANYTIME SOON
    corrections = {
        "Begon et al.": (1, 1, -1),
        "Bourne et al.": (-1, -1, -1),
        "Moissenet et al.": (1, -1, 0),
        "Oki et al.": (-1, -1, 1),
        "Matsumura et al.": (-1, -1, 1),
    }

    for article, correction in corrections.items():
        condition = sub_df["article"] == article
        dof1 = sub_df["degree_of_freedom"] == 1
        dof2 = sub_df["degree_of_freedom"] == 2
        dof3 = sub_df["degree_of_freedom"] == 3
        sub_df.loc[condition & dof1, "value"] *= correction[0]
        sub_df.loc[condition & dof2, "value"] *= correction[1]
        sub_df.loc[condition & dof3, "value"] *= correction[2]

    return sub_df


def main():
    export = True
    mvt = "frontal plane elevation"
    df = import_data(correction=True)
    sub_df = df[df["humeral_motion"] == mvt]
    sub_df = manual_corrections(sub_df)
    dfi = DataFrameInterface(sub_df)
    plt = DataPlanchePlotting(dfi, restrict_to_joints=["scapulothoracic"])
    plt.plot()
    plt.update_style()
    plt.show()

    if export:
        plt.fig.write_image(f"../plots/{mvt}_scapulothoracic.png")
        plt.fig.write_image(f"../plots/{mvt}_scapulothoracic.pdf")
        plt.fig.write_html(f"../plots/{mvt}_scapulothoracic.html", include_mathjax="cdn")


if __name__ == "__main__":
    main()
