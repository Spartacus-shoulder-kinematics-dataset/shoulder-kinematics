"""
🔄 The YXY Euler Angle Conundrum: Navigating Rotational Ambiguities 🧭

This script explores the intricacies of converting between rotation matrices and YXY Euler angles,
addressing a common pitfall in many matrix-to-Euler converters.

Key Concepts:
1. YXY Euler Angle Sequence: R = Ry(α) * Rx(β) * Ry(γ)
   where α, γ ∈ [-π, π] and traditionally β ∈ [0, π]
R = [cos(α)cos(γ) - sin(α)cos(β)sin(γ)    sin(α)sin(β)    cos(α)sin(γ) + sin(α)cos(β)cos(γ)]
    [sin(β)sin(γ)                         cos(β)          -sin(β)cos(γ)                    ]
    [-sin(α)cos(γ) - cos(α)cos(β)sin(γ)   cos(α)sin(β)    -sin(α)sin(γ) + cos(α)cos(β)cos(γ)]

2. The Fundamental Issue:
   For a rotation matrix R = [rij], the standard conversion typically uses:
   β = arccos(r22)
   α = atan2(r12, r32)
   γ = atan2(r21, -r23)

   However, this approach fails to distinguish between β and -β, as cos(-β) = cos(β).

3. The Ambiguity:
   (α, β, γ) and (α ± π, -β, γ ± π) can represent the same rotation.

4. Our Solution:
   We introduce a check based on sin(β):
   If r21 < 0 or r23 > 0, we infer β < 0 and adjust accordingly.

This script demonstrates:
- How standard converters can produce inconsistent results
- A method to resolve the β sign ambiguity
- Proper normalization of angles to ensure [-π, π] range

Remember: In the realm of 3D rotations, not all paths lead to Rome,
but they might lead to the same orientation! 🌐

Let's unravel this rotational riddle! 🧩
"""

import biorbd
import numpy as np

from spartacus.src.corrections.angle_conversion_callbacks import from_euler_angles_to_rotation_matrix, mat_2_rotation

alpha = 0.1
beta = 0.3
# beta = 0.3 + np.pi / 2
# beta = 0.3 + np.pi
gamma = 0.4
seq = "yxy"

print("\nPositive beta")
rot_matrix = from_euler_angles_to_rotation_matrix(seq, alpha, beta, gamma)
print(rot_matrix)
print("sin(beta) * sin(gamma) =     ", np.sin(beta) * np.sin(gamma))
print("cos(beta) =                  ", np.cos(beta))
print("- sin(beta) * cos(gamma) =   ", -np.sin(beta) * np.cos(gamma))
print("Condition to detect sign of beta: sin(beta) * sin(gamma) < 0 or -sin(beta) * cos(gamma) > 0")
print("condition 1: ", np.sin(beta) * np.sin(gamma) < 0)
print("condition 2: ", -np.sin(beta) * np.cos(gamma) > 0)
mat_object = mat_2_rotation(rot_matrix)
new_angles = biorbd.Rotation.toEulerAngles(mat_object, seq=seq).to_array()
print("Biorbd angles :", new_angles)
print("new rotation matrix:\n", from_euler_angles_to_rotation_matrix(seq, new_angles[0], new_angles[1], new_angles[2]))

# Test when the value is negative
beta *= -1
print("\nNegative beta")
rot_matrix = from_euler_angles_to_rotation_matrix(seq, alpha, beta, gamma)
print(rot_matrix)
print("sin(beta) * sin(gamma) =     ", np.sin(beta) * np.sin(gamma))
print("cos(beta) =                  ", np.cos(beta))
print("- sin(beta) * cos(gamma) =   ", -np.sin(beta) * np.cos(gamma))
print("Condition to detect sign of beta: sin(beta) * sin(gamma) < 0 or -sin(beta) * cos(gamma) > 0")
print("condition 1: ", np.sin(beta) * np.sin(gamma) < 0)
print("condition 2: ", -np.sin(beta) * np.cos(gamma) > 0)
mat_object = mat_2_rotation(rot_matrix)
new_angles = biorbd.Rotation.toEulerAngles(mat_object, seq=seq).to_array()
print("Biorbd angles :", new_angles)
print("new rotation matrix:\n", from_euler_angles_to_rotation_matrix(seq, new_angles[0], new_angles[1], new_angles[2]))
if rot_matrix[1, 0] < 0 or rot_matrix[1, 2] > 0:
    new_angles[1] = -new_angles[1]
    new_angles[0] += np.pi
    new_angles[2] += np.pi

print("Biorbd angles fixed:", new_angles)
print("new rotation matrix:\n", from_euler_angles_to_rotation_matrix(seq, new_angles[0], new_angles[1], new_angles[2]))

new_angles[0] = (new_angles[0] + np.pi) % (2 * np.pi) - np.pi  # α in [-π, π]
new_angles[1] = (new_angles[1] + np.pi) % (2 * np.pi) - np.pi  # β in [-π, π]
new_angles[2] = (new_angles[2] + np.pi) % (2 * np.pi) - np.pi  # γ in [-π, π]

print("Biorbd angles fixed and normalized:", new_angles)
print("new rotation matrix:\n", from_euler_angles_to_rotation_matrix(seq, new_angles[0], new_angles[1], new_angles[2]))
