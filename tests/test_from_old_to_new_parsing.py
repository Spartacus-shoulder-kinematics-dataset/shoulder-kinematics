import numpy as np
import pandas as pd

from spartacus import DatasetCSV, Segment, BiomechDirection, BiomechCoordinateSystem, AnatomicalLandmark
from spartacus.src.checks import (
    check_segment_filled_with_nan,
    check_is_isb_segment,
    check_is_isb_correctable,
)
from spartacus.src.frame_reader import Frame
from spartacus.src.utils import (
    get_segment_columns,
    get_segment_columns_direction,
)

print_warnings = True


to_pass_because_left_side = [
    ("Begon et al.", Segment.CLAVICLE),  # clavicle
    ("Begon et al.", Segment.SCAPULA),  # scapula
    ("Begon et al.", Segment.HUMERUS),  # humerus
    ("Begon et al.", Segment.THORAX),  # thorax
]
to_pass_because_geometric = [
    ("Fung et al.", Segment.THORAX),  # geometric Thorax feature implement SoloVectors
    ("Fung et al.", Segment.HUMERUS),  # geometric Humerus feature implement SoloVectors
    ("Fung et al.", Segment.SCAPULA),  # geometric Scapula feature implement SoloVectors
    (
        "Gutierrez Delgado et al.",
        Segment.CLAVICLE,
    ),  # geometric ?? is it correct on clavicle doest seem consistent with the figure.
    (
        "Moissenet et al.",
        Segment.CLAVICLE,
    ),  # geometric clavicle ?? is it correct on clavicle doest seem consistent with the figure ?
    ("Kijima et al.", Segment.HUMERUS),
    ("Kijima et al.", Segment.SCAPULA),
    ("Kim et al.", Segment.HUMERUS),
    ("Kim et al.", Segment.SCAPULA),
    ("Matsuki et al.", Segment.HUMERUS),
    ("Matsuki et al.", Segment.SCAPULA),
    ("Nishinaka et al.", Segment.SCAPULA),
    ("Sahara et al.", Segment.SCAPULA),
    ("Sugi et al.", Segment.HUMERUS),
    ("Sugi et al.", Segment.SCAPULA),
]
to_pass_because_not_filled = [
    ("Henninger et al.", Segment.THORAX),
    ("Henninger et al.", Segment.HUMERUS),
    ("Henninger et al.", Segment.SCAPULA),
    ("Henninger et al.", Segment.CLAVICLE),
]  # thorax
to_pass_because_thorax_is_imaging_system = [
    ("Kijima et al.", Segment.THORAX),  # thorax
    ("Kim et al.", Segment.THORAX),  # thorax
    ("Kozono et al.", Segment.THORAX),  # thorax
    ("Matsuki et al.", Segment.THORAX),  # thorax
    ("Nishinaka et al.", Segment.THORAX),  # thorax
    ("Sahara et al.", Segment.THORAX),  # thorax
    ("Sugi et al.", Segment.THORAX),  # thorax
]  # thorax
to_pass_because_there_is_mislabel = [
    ("Ludewig et al.", Segment.CLAVICLE),  # clavicle z^y_thorax* should be replaced by y_thorax*^z
    ("Oki et al.", Segment.CLAVICLE),  # clavicle z^y_thorax* should be replaced by y_thorax*^z
    ("Teece et al.", Segment.THORAX),  # thorax y^z should be replaced by z^x ??
    ("Fung et al.", Segment.CLAVICLE),  # -x_thorax* ???
    ("Matsuki et al.", Segment.CLAVICLE),  # y_thorax^z ? instead of z^y_thorax
    ("Sahara et al.", Segment.CLAVICLE),  # y_thorax^z ? instead of z^y_thorax
    (
        "Teece et al.",
        Segment.CLAVICLE,
    ),  # y_thorax^z ? instead of z^y_thorax and y^z instead of x^y ?? # Not display on the figure I have
]


def test_new_parsing():
    df = pd.read_csv(DatasetCSV.CLEAN.value)
    df = df.where(pd.notna(df), None)
    author_ok = []
    for i, row in df.iterrows():
        print(row.dataset_authors)
        count = 0
        for segment_enum in Segment:
            print(segment_enum)
            tuple_test = (row.dataset_authors, segment_enum)
            if row.dataset_authors == "Fung et al.":
                print("hey")
            if (
                tuple_test in to_pass_because_left_side
                or tuple_test in to_pass_because_geometric
                or tuple_test in to_pass_because_not_filled
                or tuple_test in to_pass_because_thorax_is_imaging_system
                or tuple_test in to_pass_because_there_is_mislabel
            ):
                count += 1
                continue

            segment_cols = get_segment_columns(segment_enum)
            segment_cols_direction = get_segment_columns_direction(segment_enum)
            # first check
            if check_segment_filled_with_nan(row, segment_cols, print_warnings=print_warnings):
                continue

            frame = Frame.from_xyz_string(
                x_axis=row[segment_cols_direction[0]],
                y_axis=row[segment_cols_direction[1]],
                z_axis=row[segment_cols_direction[2]],
                origin=row[segment_cols_direction[3]],
                segment=segment_enum,
            )

            assert frame.x_axis.biomech_direction() == BiomechDirection.from_string(row[segment_cols[0]])
            assert frame.y_axis.biomech_direction() == BiomechDirection.from_string(row[segment_cols[1]])
            assert frame.z_axis.biomech_direction() == BiomechDirection.from_string(row[segment_cols[2]])

            # build the coordinate system
            bsys = BiomechCoordinateSystem.from_biomech_directions(
                x=BiomechDirection.from_string(row[segment_cols[0]]),
                y=BiomechDirection.from_string(row[segment_cols[1]]),
                z=BiomechDirection.from_string(row[segment_cols[2]]),
                origin=AnatomicalLandmark.from_string(row[segment_cols[3]]),
                segment=segment_enum,
            )
            # second check
            if not check_is_isb_segment(row, bsys, print_warnings=print_warnings):
                output = False

            if not check_is_isb_correctable(row, bsys, print_warnings=print_warnings):
                output = False

            # if not check_correction_methods(self, bsys, print_warnings=print_warnings):
            #     output = False

            # third check if the segment is direct or not
            if not bsys.is_direct():
                if print_warnings:
                    print(
                        f"{row.dataset_authors}, " f"Segment {segment_enum.value} is not direct, " f"it should be !!!"
                    )
                output = False
        if count == 0:
            author_ok.append(row.dataset_authors)

    print(np.unique(author_ok))
