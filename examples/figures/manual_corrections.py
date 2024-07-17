from spartacus import import_data, DataFrameInterface, DataPlanchePlotting


def manual_corrections(sub_df):
    # ONLY FOR DISPLAYING PURPOSE, EXPECT THIS TO DISAPPEAR ANYTIME SOON
    # DONE FOR GH ELEVATION.
    corrections_glenohumeral = {
        # "Henninger et al.": (1, -1, -1),
        # "Yoshida et al.": (1, -1, -1),
    }

    corrections_scapulothoracic = {
        "Begon et al.": (1, 1, -1),
        "Bourne et al.": (-1, -1, -1),
        "Moissenet et al.": (1, -1, 1),
        "Oki et al.": (-1, -1, 1),
        "Matsumura et al.": (-1, -1, 1),
    }

    # Signs ok to me
    corrections_sternoclavicular = {
        "Oki et al.": (-1, -1, 1),
    }
    # Signs ok to me
    corrections_acromioclavicular = {
        "Begon et al.": (-1, 1, 1),
    }

    sub_df = apply_correction(sub_df, joint="glenohumeral", corrections=corrections_glenohumeral)
    sub_df = apply_correction(sub_df, joint="scapulothoracic", corrections=corrections_scapulothoracic)
    sub_df = apply_correction(sub_df, joint="sternoclavicular", corrections=corrections_sternoclavicular)
    sub_df = apply_correction(sub_df, joint="acromioclavicular", corrections=corrections_acromioclavicular)

    specific_correction = {
        "Moissenet et al.": (
            lambda value: value * -1 if value > 0 else value,
            lambda value: value,
            lambda value: value * -1 if value > 0 else value,
        )
    }
    sub_df = apply_specific_correction(sub_df, joint="glenohumeral", corrections=specific_correction)

    return sub_df


def apply_specific_correction(sub_df, joint, corrections):
    for article, correction in corrections.items():
        joint_condition = sub_df["joint"] == joint
        condition = sub_df["article"] == article
        dof1 = sub_df["degree_of_freedom"] == 1
        dof2 = sub_df["degree_of_freedom"] == 2
        dof3 = sub_df["degree_of_freedom"] == 3

        sub_df.loc[joint_condition & condition & dof1, "value"] = sub_df.loc[
            joint_condition & condition & dof1, "value"
        ].apply(correction[0])

        sub_df.loc[joint_condition & condition & dof2, "value"] = sub_df.loc[
            joint_condition & condition & dof2, "value"
        ].apply(correction[1])

        sub_df.loc[joint_condition & condition & dof3, "value"] = sub_df.loc[
            joint_condition & condition & dof3, "value"
        ].apply(correction[2])

    return sub_df


def apply_correction(sub_df, joint, corrections):
    for article, correction in corrections.items():
        joint_condition = sub_df["joint"] == joint
        condition = sub_df["article"] == article
        dof1 = sub_df["degree_of_freedom"] == 1
        dof2 = sub_df["degree_of_freedom"] == 2
        dof3 = sub_df["degree_of_freedom"] == 3
        sub_df.loc[joint_condition & condition & dof1, "value"] *= correction[0]
        sub_df.loc[joint_condition & condition & dof2, "value"] *= correction[1]
        sub_df.loc[joint_condition & condition & dof3, "value"] *= correction[2]

    return sub_df


def main():
    export = False
    mvt = "sagittal elevation"
    df = import_data(correction=True)
    sub_df = df[df["humeral_motion"] == mvt]
    sub_df = sub_df[sub_df["joint"] == "glenohumeral"]
    sub_df = manual_corrections(sub_df)
    dfi = DataFrameInterface(sub_df)
    plt = DataPlanchePlotting(dfi)
    plt.plot()
    plt.update_style()
    plt.show()

    if export:
        plt.fig.write_image(f"../plots/{mvt}.png")
        plt.fig.write_image(f"../plots/{mvt}.pdf")
        plt.fig.write_html(f"../plots/{mvt}.html", include_mathjax="cdn")


if __name__ == "__main__":
    main()
