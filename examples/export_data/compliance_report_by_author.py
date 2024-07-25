import numpy as np
import pandas as pd

from spartacus import DatasetCSV, Segment
from spartacus import JointType
from spartacus.src.checks import check_segment_filled_with_nan
from spartacus.src.compliance import JointCompliance, SegmentCompliance
from spartacus.src.utils import get_segment_columns
from spartacus.src.utils_setters import (
    set_joint_from_row,
    set_thoracohumeral_angle_from_row,
    set_parent_segment_from_row,
    set_child_segment_from_row,
)


def main():
    print_warnings = True

    df = pd.read_csv(DatasetCSV.CLEAN.value)
    df = df.where(pd.notna(df), None)
    authors = df["dataset_authors"].unique().tolist()

    df_deviation = pd.DataFrame(
        columns=[
            "id",
            "dataset_authors",
            "thorax_c1",
            "thorax_c2",
            "thorax_c3",
            "clavicle_c1",
            "clavicle_c2",
            "clavicle_c3",
            "scapula_c1",
            "scapula_c2",
            "scapula_c3",
            "humerus_c1",
            "humerus_c2",
            "humerus_c3",
            "sternoclavicular_c4",
            "sternoclavicular_c5",
            "acromioclavicular_c4",
            "acromioclavicular_c5",
            "scapulothoracic_c4",
            "scapulothoracic_c5",
            "glenohumeral_c4",
            "glenohumeral_c5",
            "thoracohumeral_c6",
        ]
    )

    # collect joint for which I have data
    joint_data = {author: [] for author in authors}
    df_grouped = df.groupby("dataset_authors")["joint"].agg(lambda x: list(set(x))).reset_index()
    joints_per_author = df_grouped.set_index("dataset_authors")["joint"].to_dict()

    for i, author in enumerate(authors):
        print(f"Processing {author} ({i + 1}/{len(authors)})")

        subdf = df[df["dataset_authors"] == author]
        first_row = subdf.iloc[0]

        dico_d = {}
        dico_d["id"] = first_row["dataset_id"]
        dico_d["dataset_authors"] = first_row["dataset_authors"]

        for segment in Segment:

            segment_cols = get_segment_columns(segment)
            if not check_segment_filled_with_nan(first_row, segment_cols, print_warnings=print_warnings):
                bsys_segment = set_parent_segment_from_row(first_row, segment)
                compliance = SegmentCompliance(mode="rotation", bsys=bsys_segment)
                dico_d[f"{segment.to_string}_c1"] = 0 if compliance.is_c1 else 1
                dico_d[f"{segment.to_string}_c2"] = 0 if compliance.is_c2 else 1
                dico_d[f"{segment.to_string}_c3"] = 0 if compliance.is_c3 else 1
            if (
                check_segment_filled_with_nan(first_row, segment_cols, print_warnings=print_warnings)
                and first_row[segment_cols[3]] is not None
            ):
                #  for nishinaka for example that only has translational information
                bsys_segment = set_child_segment_from_row(first_row, segment)
                deviation = SegmentCompliance(mode="rotation", bsys=bsys_segment)
                dico_d[f"{segment.to_string}_c3"] = 0 if deviation.is_c3 else 1

        for joint_type_str in joints_per_author[author]:
            joint_type = JointType.from_string(joint_type_str)
            # first row of that joint
            first_row = subdf[subdf["joint"] == joint_type_str].iloc[0]
            joint = set_joint_from_row(first_row, joint_type)

            rotation_parent_deviation = SegmentCompliance(mode="rotation", bsys=joint.parent_segment)
            rotation_child_deviation = SegmentCompliance(mode="rotation", bsys=joint.child_segment)

            thoracohumeral_angle = set_thoracohumeral_angle_from_row(first_row)
            rotation_joint_deviation = JointCompliance(
                mode="rotation", joint=joint, thoracohumeral_angle=thoracohumeral_angle
            )
            # missing a if for nishinaka
            dico_d[f"{joint_type.to_string}_c4"] = 0 if rotation_joint_deviation.is_c4 else 1
            if joint.translation_origin is not None:
                dico_d[f"{joint_type.to_string}_c5"] = 0 if rotation_joint_deviation.is_c5 else 1
            dico_d[f"thoracohumeral_c6"] = 0 if rotation_joint_deviation.is_c6 else 1

        df_deviation = pd.concat([df_deviation, pd.DataFrame([dico_d])], ignore_index=True)

    return df_deviation.replace(np.nan, "-")


if __name__ == "__main__":
    df = main()
    df.to_csv("deviation_table.csv")
    print(df.values)
