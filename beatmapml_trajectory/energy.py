import numpy as np
import math

__all__ = [
    'make_energy'
]

EPSILON = 1e-6


def make_energy(trajectory: np.ndarray) -> np.array:
    """Figure out energy consumed in the trajectory

    Args:
        trajectory (np.ndarray): The trajectory produced by make_inputs.
        capture_rate (int): The capture rate of the trajectory in Hz

    Returns:
        Energy of the trajectory in a lenght L 1d array
    """
    angle = trajectory[:, 0] * math.pi
    velocity = trajectory[:, 1] * [np.cos(angle), np.sin(angle)]
    accel = np.diff(velocity)
    accel /= np.linalg.norm(accel, ord=2, axis=0) + EPSILON
    v1_proj = velocity[0, :-1] * accel[0] + velocity[1, :-1] * accel[1]
    v2_proj = velocity[0, 1:] * accel[0] + velocity[1, 1:] * accel[1]
    energy = np.zeros((trajectory.shape[0],), dtype=np.float32)
    flip_dir = (v1_proj * v2_proj) < 0
    energy[:-1][flip_dir] = v1_proj[flip_dir] ** 2 + v2_proj[flip_dir] ** 2
    energy[:-1][~flip_dir] = (v1_proj[~flip_dir] - v2_proj[~flip_dir]) ** 2
    return energy
