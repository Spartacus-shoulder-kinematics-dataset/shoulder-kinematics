import numpy as np
import pytest

from spartacus.src.corrections.kolz_matrices import get_kolz_rotation_matrix
from spartacus.src.enums_biomech import Correction


def test_kolz():
    R = get_kolz_rotation_matrix(Correction.SCAPULA_KOLZ_AC_TO_PA_ROTATION, orthonormalize=False)

    assert R.shape == (3, 3)
    assert R[0, 0] == 0.964732483180704
    assert R[1, 0] == -0.0568391733041278
    assert R[2, 0] == 0.257022458695919
    assert R[0, 1] == 0.0100557410449752
    assert R[1, 1] == 0.983654375192805
    assert R[2, 1] == 0.17978585104532
    assert R[0, 2] == -0.263040145164248
    assert R[1, 2] == -0.170860699232318
    assert R[2, 2] == 0.949534887979275

    R = get_kolz_rotation_matrix(Correction.SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION, orthonormalize=False)

    assert R.shape == (3, 3)
    assert R[0, 0] == 0.949221220989932
    assert R[0, 1] == 0.0563324947886947
    assert R[0, 2] == 0.309524996814902
    assert R[1, 0] == 0.00968382059265423
    assert R[1, 1] == 0.978141286964525
    assert R[1, 2] == -0.207715782631251
    assert R[2, 0] == -0.314460326974823
    assert R[2, 1] == 0.200165613346213
    assert R[2, 2] == 0.927926952940059

    R = get_kolz_rotation_matrix(Correction.SCAPULA_KOLZ_AC_TO_PA_ROTATION, orthonormalize=True)
    assert R.shape == (3, 3)

    np.testing.assert_almost_equal(R[0, 0], 0.96473248)
    np.testing.assert_almost_equal(R[0, 1], 0.01005574)
    np.testing.assert_almost_equal(R[0, 2], -0.26304015)
    np.testing.assert_almost_equal(R[1, 0], -0.05683917)
    np.testing.assert_almost_equal(R[1, 1], 0.98365438)
    np.testing.assert_almost_equal(R[1, 2], -0.1708607)
    np.testing.assert_almost_equal(R[2, 0], 0.25702246)
    np.testing.assert_almost_equal(R[2, 1], 0.17978585)
    np.testing.assert_almost_equal(R[2, 2], 0.94953489)
    np.testing.assert_almost_equal(np.linalg.det(R), 1.0)

    with pytest.raises(ValueError):
        get_kolz_rotation_matrix(correction=Correction.TO_ISB_ROTATION)
