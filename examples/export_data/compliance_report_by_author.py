import numpy as np
import pandas as pd

from spartacus import JointType, Segment
from spartacus import Spartacus
from spartacus.src.checks import check_segment_filled_with_nan
from spartacus.src.compliance import JointCompliance, SegmentCompliance
from spartacus.src.utils import get_segment_columns_direction
from spartacus.src.utils_setters import (
    set_joint_from_row,
    set_thoracohumeral_angle_from_row,
    set_parent_segment_from_row,
    set_child_segment_from_row,
)


def main():
    print_warnings = True

    sp = Spartacus.load(check_and_import=False, exclude_dataset_without_series=False)
    df = sp.dataframe

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

            segment_cols = get_segment_columns_direction(segment)
            if not check_segment_filled_with_nan(first_row, segment_cols, print_warnings=print_warnings):
                bsys_segment = set_parent_segment_from_row(first_row, segment)
                compliance = SegmentCompliance(bsys=bsys_segment)
                dico_d[f"{segment.to_string}_c1"] = compliance.is_c1
                dico_d[f"{segment.to_string}_c2"] = compliance.is_c2
                dico_d[f"{segment.to_string}_c3"] = compliance.is_c3
            if (
                check_segment_filled_with_nan(first_row, segment_cols, print_warnings=print_warnings)
                and first_row[segment_cols[3]] is not None
            ):
                #  for nishinaka for example that only has translational information
                bsys_segment = set_child_segment_from_row(first_row, segment)
                deviation = SegmentCompliance(bsys=bsys_segment)
                dico_d[f"{segment.to_string}_c3"] = deviation.is_c3

        for joint_type_str in joints_per_author[author]:
            joint_type = JointType.from_string(joint_type_str)
            # first row of that joint
            first_row = subdf[subdf["joint"] == joint_type_str].iloc[0]
            joint = set_joint_from_row(first_row, joint_type)

            thoracohumeral_angle = set_thoracohumeral_angle_from_row(first_row)
            joint_deviation = JointCompliance(joint=joint, thoracohumeral_angle=thoracohumeral_angle)

            if joint.euler_sequence is not None:
                dico_d[f"{joint_type.to_string}_c4"] = joint_deviation.is_c4
            if joint.translation_origin is not None:
                dico_d[f"{joint_type.to_string}_c5"] = joint_deviation.is_c5
            dico_d[f"thoracohumeral_c6"] = joint_deviation.is_c6

        df_deviation = pd.concat([df_deviation, pd.DataFrame([dico_d])], ignore_index=True)

    return df_deviation.replace(np.nan, "-")


if __name__ == "__main__":
    df = main()
    df.to_csv("deviation_table.csv")
    print(df.values)
