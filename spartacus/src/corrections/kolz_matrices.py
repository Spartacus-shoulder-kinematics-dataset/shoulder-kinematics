import numpy as np
from ..enums_biomech import Correction


def get_kolz_rotation_matrix(correction: Correction, orthonormalize: bool = True) -> np.ndarray:
    """
    This function returns the rotation matrix for the given correction.

    Parameters
    ----------
    correction : Correction
        The correction to apply.
    orthonormalize : bool, optional
        If True, the rotation matrix is orthonormalized. The default is True.

    Returns
    -------
    np.ndarray
        The rotation matrix for the given correction.
        R_isb_local, such that a_in_isb = R_isb_local * a_in_local

    Source
    ------
    Kolz, C. W., Sulkar, H. J., Aliaj, K., Tashjian, R. Z., Chalmers, P. N., Qiu, Y., ... & Henninger, H. B. (2020).
    Reliable interpretation of scapular kinematics depends on coordinate system definition. Gait & posture, 81, 183-190.
    """

    if correction == Correction.SCAPULA_KOLZ_AC_TO_PA_ROTATION:
        # meaning a vector in AC coordinate system is expressed as:
        # a' = R * a
        # where a' is the vector expressed in the PA coordinate system
        R_pa_ac = np.array(
            [
                [0.964732483180704, 0.0100557410449752, -0.263040145164248],
                [-0.0568391733041278, 0.983654375192805, -0.170860699232318],
                [0.257022458695919, 0.17978585104532, 0.949534887979275],
            ],
        )
        R = R_pa_ac

    elif correction == Correction.SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION:
        # meaning a vector in glenoid coordinate system is expressed as:
        # a' = R * a
        # where a' is the vector expressed in the PA coordinate system
        R_gc_pa = np.array(
            [
                [0.949221220989932, 0.00968382059265423, -0.314460326974823],
                [0.0563324947886947, 0.978141286964525, 0.200165613346213],
                [0.309524996814902, -0.207715782631251, 0.927926952940059],
            ]
        )
        R_pa_gc = R_gc_pa.T
        R = R_pa_gc

    else:
        raise ValueError(
            f"{correction} is not a valid correction. Only {Correction.SCAPULA_KOLZ_AC_TO_PA_ROTATION} "
            f"and {Correction.SCAPULA_KOLZ_GLENOID_TO_PA_ROTATION} are valid corrections."
        )

    return orthonormalize_matrix(R) if orthonormalize else R


#
def orthonormalize_matrix(matrix):
    """Normalize a matrix with svd and return the normalized matrix"""
    # normalize the matrix
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    normalized_matrix = u @ vh

    # print("matrix")
    # # print in scientific notation with e
    # np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.3e}'.format})
    # print(matrix)
    # print(np.linalg.det(matrix))
    #
    # print("normalized matrix")
    # print(normalized_matrix)
    # print(np.linalg.det(normalized_matrix))

    # print("differences between matrix and normalized matrix")
    # print(matrix - normalized_matrix)

    return normalized_matrix
