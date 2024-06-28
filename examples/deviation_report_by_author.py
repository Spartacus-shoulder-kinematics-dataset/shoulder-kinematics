import numpy as np
import pandas as pd

from spartacus import DatasetCSV, Segment
from spartacus import JointType
from spartacus.src.checks import check_segment_filled_with_nan
from spartacus.src.deviation import SegmentDeviation, JointDeviation
from spartacus.src.utils import get_segment_columns
from spartacus.src.utils_setters import set_joint_from_row, set_thoracohumeral_angle_from_row

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


# something wrong with a peach of code yet
to_pass_again = [
    "Gutierrez Delgado et al.",
    "Kijima et al.",
    "Kim et al.",
    "Matsuki et al.",
    "Nishinaka et al.",
    "Sahara et al.",
    "Sugi et al.",
]

for i, author in enumerate(authors):
    print(f"Processing {author} ({i + 1}/{len(authors)})")

    if author in to_pass_again:
        continue

    subdf = df[df["dataset_authors"] == author]
    first_row = subdf.iloc[0]

    dico = {}
    dico["id"] = first_row["dataset_id"]
    dico["dataset_authors"] = first_row["dataset_authors"]

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
        rot_tot = (
            rotation_parent_deviation.total() * rotation_child_deviation.total() * rotation_joint_deviation.total()
        )

        joint_str = joint_type.to_string
        dico[f"{joint_str}_rotation"] = rot_tot

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
        dico[f"{joint_str}_translation"] = trans_tot

    dico["median"] = np.median([e for e in list(dico.values())[2:]])
    df_compliance = pd.concat([df_compliance, pd.DataFrame([dico])], ignore_index=True)

df_compliance.to_csv("compliance_table.csv")
print("Done")
