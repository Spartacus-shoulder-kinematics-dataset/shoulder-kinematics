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

    assert all_good.is_isb_oriented() == True
    assert all_good.is_mislabeled() == False
    assert all_good.is_any_axis_wrong_sens() == False
    assert all_good.get_segment_risk_quantification("proximal", "rotation") == 1

    mislabeled = BiomechCoordinateSystem(
        segment=Segment.THORAX,
        antero_posterior_axis=CartesianAxis.plusY,
        infero_superior_axis=CartesianAxis.plusX,
        medio_lateral_axis=CartesianAxis.plusZ,
        origin=AnatomicalLandmark.Thorax.IJ,
    )

    assert mislabeled.is_isb_oriented() == False
    assert mislabeled.is_mislabeled() == True
    assert mislabeled.is_any_axis_wrong_sens() == False
    assert mislabeled.get_segment_risk_quantification("proximal", "rotation") == 0.9

    wrong_sens = BiomechCoordinateSystem(
        segment=Segment.THORAX,
        antero_posterior_axis=CartesianAxis.minusX,
        infero_superior_axis=CartesianAxis.minusY,
        medio_lateral_axis=CartesianAxis.minusZ,
        origin=AnatomicalLandmark.Thorax.IJ,
    )
    assert wrong_sens.is_isb_oriented() == False
    assert wrong_sens.is_mislabeled() == False
    assert wrong_sens.is_any_axis_wrong_sens() == True
    assert wrong_sens.get_segment_risk_quantification("proximal", "rotation") == 0.9

    mislabeled_and_wrong_sens = BiomechCoordinateSystem(
        segment=Segment.THORAX,
        antero_posterior_axis=CartesianAxis.minusY,
        infero_superior_axis=CartesianAxis.plusZ,
        medio_lateral_axis=CartesianAxis.plusX,
        origin=AnatomicalLandmark.Thorax.IJ,
    )

    assert mislabeled_and_wrong_sens.is_isb_oriented() == False
    assert mislabeled_and_wrong_sens.is_mislabeled() == True
    assert mislabeled_and_wrong_sens.is_any_axis_wrong_sens() == True
    assert mislabeled_and_wrong_sens.get_segment_risk_quantification("proximal", "rotation") == 0.9 * 0.9
