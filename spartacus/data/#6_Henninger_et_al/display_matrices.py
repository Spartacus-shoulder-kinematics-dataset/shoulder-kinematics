"""
This script explores the different rotation matrices of Kolz et al. article in order to make sure we understand their
rotation matrix convention.
"""

import matplotlib.pyplot as plt
import numpy as np

rotation_matrix_ac_to_gc = np.array(
    [
        [0.998564988941296, 0.00949439421388125, 0.0527050219540931],
        [0.0112709225847591, 0.924857019102618, -0.380147945569494],
        [-0.0523538839510344, 0.380196463285931, 0.923422828470312],
    ]
)
rotation_matrix_gc_to_ac = rotation_matrix_ac_to_gc.T
rotation_matrix_ac_to_pa = np.array(
    [
        [0.964732483180704, 0.0100557410449752, -0.263040145164248],
        [-0.0568391733041278, 0.983654375192805, -0.170860699232318],
        [0.257022458695919, 0.17978585104532, 0.949534887979275],
    ]
)
rotation_matrix_pa_to_ac = rotation_matrix_ac_to_pa.T
rotation_matrix_gc_to_pa = np.array(
    [
        [0.949221220989932, 0.00968382059265423, -0.314460326974823],
        [0.0563324947886947, 0.978141286964525, 0.200165613346213],
        [0.309524996814902, -0.207715782631251, 0.927926952940059],
    ]
)
rotation_matrix_pa_to_gc = rotation_matrix_gc_to_pa.T

np.testing.assert_allclose(np.linalg.det(rotation_matrix_ac_to_gc), 1)
np.testing.assert_allclose(np.linalg.det(rotation_matrix_gc_to_ac), 1)
np.testing.assert_allclose(np.linalg.det(rotation_matrix_ac_to_pa), 1)


def rotation_matrix_2_xyz(rotation_matrix):
    x = rotation_matrix[:, 0]
    y = rotation_matrix[:, 1]
    z = rotation_matrix[:, 2]
    return x, y, z


def plot_frame(ax, x, y, z, linestyle="solid"):
    ax.plot([0, x[0]], [0, x[1]], [0, x[2]], color="r", linestyle=linestyle)
    ax.plot([0, y[0]], [0, y[1]], [0, y[2]], color="g", linestyle=linestyle)
    ax.plot([0, z[0]], [0, z[1]], [0, z[2]], color="b", linestyle=linestyle)

    return ax


# three subplots horizontally
ax = plt.subplot(1, 3, 1, projection="3d")

ax.set_title("Acromioclavicular basis as a reference")
ax = plot_frame(ax, [1, 0, 0], [0, 1, 0], [0, 0, 1], linestyle="solid")
x, y, z = rotation_matrix_2_xyz(rotation_matrix_gc_to_ac.T)
# x, y, z = rotation_matrix_2_xyz(rotation_matrix_ac_to_gc)
ax = plot_frame(ax, x, y, z, linestyle="dashed")
x, y, z = rotation_matrix_2_xyz(rotation_matrix_pa_to_ac.T)
# x, y, z = rotation_matrix_2_xyz(rotation_matrix_ac_to_pa)
ax = plot_frame(ax, x, y, z, linestyle="dashdot")

ax.plot([0, 0], [0, 0], [0, 0], color="k", linestyle="solid", label="acromioclavicular")
ax.plot([0, 0], [0, 0], [0, 0], color="k", linestyle="dashed", label="glenohumeral")
ax.plot([0, 0], [0, 0], [0, 0], color="k", linestyle="dashdot", label="posterior aspect")

plt.legend()

ax.grid(False)

# it displays as expected, looking at the z axis (blue), solid lines are over the dashed lines
# considering a_in_GC vector in the GC frame,
# R_ac_gc = rotation_matrix_gc_to_ac.T
# R_ac_gc = rotation_matrix_ac_to_gc
# a_in_AC = R_ac_gc @ a_in_GC

ax = plt.subplot(1, 3, 2, projection="3d")
ax.set_title("Glenohumeral basis as a reference")

ax = plot_frame(ax, [1, 0, 0], [0, 1, 0], [0, 0, 1], linestyle="dashed")
x, y, z = rotation_matrix_2_xyz(rotation_matrix_ac_to_gc.T)
ax = plot_frame(ax, x, y, z, linestyle="solid")
x, y, z = rotation_matrix_2_xyz(rotation_matrix_pa_to_gc.T)
ax = plot_frame(ax, x, y, z, linestyle="dashdot")

ax.plot([0, 0], [0, 0], [0, 0], color="r", linestyle="solid", label="x")
ax.plot([0, 0], [0, 0], [0, 0], color="g", linestyle="solid", label="y")
ax.plot([0, 0], [0, 0], [0, 0], color="b", linestyle="solid", label="z")

plt.legend()
# grid off
ax.grid(False)


ax = plt.subplot(1, 3, 3, projection="3d")
ax.set_title("Posterior aspect as a reference")

ax = plot_frame(ax, [1, 0, 0], [0, 1, 0], [0, 0, 1], linestyle="dashdot")
x, y, z = rotation_matrix_2_xyz(rotation_matrix_gc_to_pa.T)
ax = plot_frame(ax, x, y, z, linestyle="dashed")
x, y, z = rotation_matrix_2_xyz(rotation_matrix_ac_to_pa.T)
ax = plot_frame(ax, x, y, z, linestyle="solid")
ax.grid(False)

plt.show()
