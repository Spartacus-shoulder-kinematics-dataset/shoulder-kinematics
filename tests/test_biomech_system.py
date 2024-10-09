from spartacus.src.biomech_system import BiomechCoordinateSystem

from spartacus.src.enums_biomech import CartesianAxis, AnatomicalLandmark, Segment


def test_risk_routine():
    # create a biomech system
    all_good = BiomechCoordinateSystem(
        segment=Segment.THORAX,
        antero_posterior_axis=CartesianAxis.plusX,
        infero_superior_axis=CartesianAxis.plusY,
        medio_lateral_axis=CartesianAxis.plusZ,
        origin=AnatomicalLandmark.Thorax.IJ,
    )

    assert all_good.is_isb_oriented == True

    mislabeled = BiomechCoordinateSystem(
        segment=Segment.THORAX,
        antero_posterior_axis=CartesianAxis.plusY,
        infero_superior_axis=CartesianAxis.plusX,
        medio_lateral_axis=CartesianAxis.plusZ,
        origin=AnatomicalLandmark.Thorax.IJ,
    )

    assert mislabeled.is_isb_oriented == False

    wrong_sens = BiomechCoordinateSystem(
        segment=Segment.THORAX,
        antero_posterior_axis=CartesianAxis.minusX,
        infero_superior_axis=CartesianAxis.minusY,
        medio_lateral_axis=CartesianAxis.minusZ,
        origin=AnatomicalLandmark.Thorax.IJ,
    )
    assert wrong_sens.is_isb_oriented == False

    mislabeled_and_wrong_sens = BiomechCoordinateSystem(
        segment=Segment.THORAX,
        antero_posterior_axis=CartesianAxis.minusY,
        infero_superior_axis=CartesianAxis.plusZ,
        medio_lateral_axis=CartesianAxis.plusX,
        origin=AnatomicalLandmark.Thorax.IJ,
    )

    assert mislabeled_and_wrong_sens.is_isb_oriented == False
