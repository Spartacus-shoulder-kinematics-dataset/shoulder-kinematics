"""
Rewritting the rotation matrices from the paper in a way that they can be used in python and with the better conventions.
a_in_AC = R_ac_gc @ a_in_GC
"""

import matplotlib.pyplot as plt
import numpy as np

R_gc_ac = np.array(
    [
        [0.998564988941296, 0.00949439421388125, 0.0527050219540931],
        [0.0112709225847591, 0.924857019102618, -0.380147945569494],
        [-0.0523538839510344, 0.380196463285931, 0.923422828470312],
    ]
)
R_ac_gc = R_gc_ac.T
R_pa_ac = np.array(
    [
        [0.964732483180704, 0.0100557410449752, -0.263040145164248],
        [-0.0568391733041278, 0.983654375192805, -0.170860699232318],
        [0.257022458695919, 0.17978585104532, 0.949534887979275],
    ]
)
R_ac_pa = R_pa_ac.T
R_gc_pa = np.array(
    [
        [0.949221220989932, 0.00968382059265423, -0.314460326974823],
        [0.0563324947886947, 0.978141286964525, 0.200165613346213],
        [0.309524996814902, -0.207715782631251, 0.927926952940059],
    ]
)
R_pa_gc = R_gc_pa.T

np.testing.assert_allclose(np.linalg.det(R_gc_ac), 1)
np.testing.assert_allclose(np.linalg.det(R_pa_ac), 1)
np.testing.assert_allclose(np.linalg.det(R_gc_pa), 1)
