"""
This is a draft of an example of how to load a dataset with a custom standard.
Particularly useful to load translational data that cant be corrected by Spartacus.
But this relevant in their original frame of reference.

Todo: make it work
"""

from spartacus import EulerSequence, Spartacus, CartesianAxis, AnatomicalLandmark

standard = dict(
    glenohumeral={"seq": EulerSequence.ZYX},
    scapula={
        "orientation": {
            "posteroanterior": CartesianAxis.plusX,
            "superoinferior": CartesianAxis.plusY,
            "mediolateral": CartesianAxis.plusZ,
        },
        "origin": AnatomicalLandmark.Scapula.GLENOID_CENTER,
    },
)


spartacus_dataset = Spartacus.load(
    standard=standard,
)
