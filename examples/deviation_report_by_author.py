import numpy as np
import pandas as pd

from spartacus import DatasetCSV, Segment
from spartacus import JointType
from spartacus.src.checks import check_segment_filled_with_nan
from spartacus.src.deviation import SegmentDeviation, JointDeviation
from spartacus.src.utils import get_segment_columns
from spartacus.src.utils_setters import (
    set_joint_from_row,
    set_thoracohumeral_angle_from_row,
    set_parent_segment_from_row,
)

print_warnings = True

to_pass_because_geometric = [
    ("Fung et al.", Segment.HUMERUS),  # geometric Humerus feature implement SoloVectors
    ("Kijima et al.", Segment.HUMERUS),
    ("Kim et al.", Segment.HUMERUS),
    ("Matsuki et al.", Segment.HUMERUS),
    ("Sugi et al.", Segment.HUMERUS),
]

df = pd.read_csv(DatasetCSV.CLEAN.value)
df = df.where(pd.notna(df), None)
authors = df["dataset_authors"].unique().tolist()
df_deviation = pd.DataFrame(
    columns=[
        "id",
        "dataset_authors",
        "thorax_d1",
        "thorax_d2",
        "thorax_d3",
        "thorax_d4",
        "clavicle_d1",
        "clavicle_d2",
        "clavicle_d3",
        "clavicle_d4",
        "scapula_d1",
        "scapula_d2",
        "scapula_d3",
        "scapula_d4",
        "humerus_d1",
        "humerus_d2",
        "humerus_d3",
        "humerus_d4",
        "sternoclavicular_d5",
        "sternoclavicular_d6",
        "acromioclavicular_d5",
        "acromioclavicular_d6",
        "scapulothoracic_d5",
        "scapulothoracic_d6",
        "glenohumeral_d5",
        "glenohumeral_d6",
        "thoracohumeral_d7",
    ]
)

df_compliance = pd.DataFrame(
    columns=[
        "id",
        "dataset_authors",
        "sternoclavicular_rotation",
        "sternoclavicular_translation",
        "acromioclavicular_rotation",
        "acromioclavicular_translation",
        "scapulothoracic_rotation",
        "scapulothoracic_translation",
        "glenohumeral_rotation",
        "glenohumeral_translation",
        "median",
    ]
)


for i, author in enumerate(authors):
    print(f"Processing {author} ({i + 1}/{len(authors)})")

    subdf = df[df["dataset_authors"] == author]
    first_row = subdf.iloc[0]

    dico_c = {}
    dico_c["id"] = first_row["dataset_id"]
    dico_c["dataset_authors"] = first_row["dataset_authors"]

    dico_d = {}
    dico_d["id"] = first_row["dataset_id"]
    dico_d["dataset_authors"] = first_row["dataset_authors"]

    for segment in Segment:

        if (author, segment) in to_pass_because_geometric:
            continue

        segment_cols = get_segment_columns(segment)
        if check_segment_filled_with_nan(first_row, segment_cols, print_warnings=print_warnings):
            continue
        bsys_segment = set_parent_segment_from_row(first_row, segment)
        deviation = SegmentDeviation(mode="rotation", bsys=bsys_segment)
        dico_d[f"{segment.to_string}_d1"] = deviation.is_d1
        dico_d[f"{segment.to_string}_d2"] = deviation.is_d2
        dico_d[f"{segment.to_string}_d3"] = deviation.is_d3
        dico_d[f"{segment.to_string}_d4"] = deviation.is_d4

    for joint_type in JointType:
        if joint_type == JointType.THORACO_HUMERAL:
            continue
        segment_cols = get_segment_columns(joint_type.parent)
        if check_segment_filled_with_nan(first_row, segment_cols, print_warnings=print_warnings):
            continue
        segment_cols = get_segment_columns(joint_type.child)
        if check_segment_filled_with_nan(first_row, segment_cols, print_warnings=print_warnings):
            continue
        if (author, joint_type.parent) in to_pass_because_geometric or (
            author,
            joint_type.child,
        ) in to_pass_because_geometric:
            continue

        joint = set_joint_from_row(first_row, joint_type)
        thoracohumeral_angle = set_thoracohumeral_angle_from_row(first_row, joint=joint)

        rotation_parent_deviation = SegmentDeviation(mode="rotation", bsys=joint.parent_segment)
        rotation_child_deviation = SegmentDeviation(mode="rotation", bsys=joint.child_segment)
        rotation_joint_deviation = JointDeviation(
            mode="rotation", joint=joint, thoracohumeral_angle=thoracohumeral_angle
        )

        dico_d[f"{joint_type.to_string}_d5"] = rotation_joint_deviation.is_d5
        dico_d[f"{joint_type.to_string}_d6"] = rotation_joint_deviation.is_d6
        dico_d[f"thoracohumeral_d7"] = rotation_joint_deviation.is_d7

        rot_tot = (
            rotation_parent_deviation.total() * rotation_child_deviation.total() * rotation_joint_deviation.total()
        )

        joint_str = joint_type.to_string
        dico_c[f"{joint_str}_rotation"] = rot_tot

        translation_parent_deviation = SegmentDeviation(mode="rotation", bsys=joint.parent_segment)
        translation_child_deviation = SegmentDeviation(mode="rotation", bsys=joint.child_segment)
        translation_joint_deviation = JointDeviation(
            mode="rotation", joint=joint, thoracohumeral_angle=thoracohumeral_angle
        )
        trans_tot = (
            translation_parent_deviation.total()
            * translation_child_deviation.total()
            * translation_joint_deviation.total()
        )
        dico_c[f"{joint_str}_translation"] = trans_tot

    dico_c["median"] = np.median([e for e in list(dico_c.values())[2:]])
    df_compliance = pd.concat([df_compliance, pd.DataFrame([dico_c])], ignore_index=True)
    df_deviation = pd.concat([df_deviation, pd.DataFrame([dico_d])], ignore_index=True)

df_compliance.to_csv("compliance_table.csv")
df_deviation.to_csv("deviation_table.csv")
print("Done")
