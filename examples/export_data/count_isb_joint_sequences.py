"""
This example shows how to count the number of ISB joint sequences in the dataset.
The trick is to compute the value for the previous sequence and the new sequence, and then to compare them.
"""

import pandas as pd

import spartacus as sp
from spartacus import DatasetCSV, RowData, EulerSequence
from src.corrections.angle_conversion_callbacks import convert_euler_angles_and_frames_to_isb


def main():
    datasets = pd.read_csv(DatasetCSV.DATASETS.value)
    joint_data = pd.read_csv(DatasetCSV.JOINT.value)
    maximus = sp.Spartacus(datasets=datasets, joint_data=joint_data)

    df_new = pd.DataFrame(
        columns=[
            "article",
            "joint",
            "parent",
            "child",
            "old_euler_sequence",
            "isb_euler_sequence",
            "equivalent_sequence",
            "comment",
        ]
    )

    for i, row in maximus.dataframe.iterrows():
        new_dict = dict()

        print(row.dataset_authors)
        new_dict["article"] = row.dataset_authors
        row_data = RowData(row)

        new_dict["parent"] = row.parent
        new_dict["child"] = row.child
        new_dict["comment"] = None
        new_dict["joint"] = row.joint

        if not row_data.check_joint_validity(print_warnings=False):
            print(row.dataset_authors, "joint not valid")
            new_dict["comment"] = "joint not valid"
            df_new = pd.concat([df_new, pd.DataFrame([new_dict])], ignore_index=True)
            continue
        row_data.set_segments()

        if row_data.joint.euler_sequence is None:
            new_dict["comment"] = "no euler sequence"
            df_new = pd.concat([df_new, pd.DataFrame([new_dict])], ignore_index=True)
            continue

        print("previous sequence:", row_data.joint.euler_sequence.value)
        print("new sequence:", EulerSequence.isb_from_joint_type(row_data.joint.joint_type).value)
        new_dict["old_euler_sequence"] = row_data.joint.euler_sequence.value
        new_dict["isb_euler_sequence"] = EulerSequence.isb_from_joint_type(row_data.joint.joint_type).value

        if row_data.joint.euler_sequence == EulerSequence.isb_from_joint_type(row_data.joint.joint_type):
            print(row.dataset_authors, "SAME SEQUENCE")
            new_dict["comment"] = "same sequence"

        new_angles = convert_euler_angles_and_frames_to_isb(
            previous_sequence_str=row_data.joint.euler_sequence.value,
            new_sequence_str=EulerSequence.isb_from_joint_type(row_data.joint.joint_type).to_string,
            rot1=0.2,
            rot2=0.3,
            rot3=0.4,
            bsys_parent=row_data.parent_biomech_sys,
            bsys_child=row_data.child_biomech_sys,
        )

        if round(new_angles[0], 5) == 0.2 and round(new_angles[1], 5) == 0.3 and round(new_angles[2], 5) == 0.4:
            new_dict["equivalent_sequence"] = "TRUE"
        else:
            new_dict["equivalent_sequence"] = "FALSE"
            if new_dict["comment"] == "same sequence":
                new_dict["comment"] += " but at least one segment is mislabelled"

        df_new = pd.concat([df_new, pd.DataFrame([new_dict])], ignore_index=True)

    df_new.to_csv("is_isb_joint.csv")

    return df_new


def stats(df: pd.DataFrame):

    # Calculate the overall proportion of equivalent sequences
    overall_proportion = df["equivalent_sequence"].value_counts(normalize=True) * 100
    overall_n = df["equivalent_sequence"].value_counts()

    # Calculate the proportion of equivalent sequences by joint
    proportion_by_joint = df.groupby("joint")["equivalent_sequence"].value_counts(normalize=True).unstack() * 100
    n_by_joint = df.groupby("joint")["equivalent_sequence"].value_counts().unstack()

    #  join dataframes
    overall = overall_proportion.to_frame().join(overall_n.to_frame(), lsuffix="_proportion", rsuffix="_number")
    by_joint = proportion_by_joint.join(n_by_joint, lsuffix="_proportion", rsuffix="_number")

    # Assuming "obvious True same sequences" means entries where 'old_euler_sequence' and 'isb_euler_sequence' are the same
    # and the 'equivalent_sequence' is True, let's calculate that proportion.

    # Create a new column to identify rows where old and ISB sequences are exactly the same and equivalent_sequence is True
    df.loc[:, "same_sequence_true"] = (
        (df["old_euler_sequence"] == df["isb_euler_sequence"]) & (df["equivalent_sequence"] == "TRUE")
    ).to_list()

    # Calculate the overall proportion of these specific sequences
    same_sequence_true_proportion = df["same_sequence_true"].value_counts(normalize=True) * 100
    same_sequence_true_n = df["same_sequence_true"].value_counts()

    # Calculate the proportion of these sequences by joint
    same_sequence_true_proportion_by_joint = (
        df.groupby("joint")["same_sequence_true"].value_counts(normalize=True).unstack() * 100
    )
    same_sequence_true_n_by_joint = df.groupby("joint")["same_sequence_true"].value_counts().unstack()

    same_sequence_true = same_sequence_true_proportion.to_frame().join(
        same_sequence_true_n.to_frame(), lsuffix="_proportion", rsuffix="_number"
    )
    same_sequence_true_by_joint = same_sequence_true_proportion_by_joint.join(
        same_sequence_true_n_by_joint, lsuffix="_proportion", rsuffix="_number"
    )

    # save
    overall.to_csv("equivalent_overall_proportion.csv")
    by_joint.to_csv("equivalent_by_joint_proportion.csv")
    same_sequence_true.to_csv("already_isb_true_proportion.csv")
    same_sequence_true_by_joint.to_csv("already_isb_by_joint_proportion.csv")


if __name__ == "__main__":
    df = main()

    # drop rows with "comment" that has "not valid" or "no euler sequence"
    clean_df = df[~df["comment"].str.contains("not valid|no euler sequence", na=False)]
    # print dropped rows with their article name
    excluded = df[df["comment"].str.contains("not valid|no euler sequence", na=False)][["article", "comment"]]
    # can you keep the line number associated ? yes

    excluded.to_csv("excluded_study.csv")

    stats(clean_df)

    print("DONE")
