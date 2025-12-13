import numpy as np
from typing import Tuple, Optional

def dist(
        u: np.ndarray,
        v: np.ndarray
):
    """
    Calculates the Euclidean distance between two points.

    :param u: First point given by a vector.
    :param v: Second point given by a vector.
    :return: If the distance between u and v is too small, return 0, otherwise, return the distance between u and v.
    """
    uv_path_len = np.linalg.norm(u - v)

    if uv_path_len < 1e-9:
        return 0.0

    return float(uv_path_len)

def get_distractor_sphere(
        env
) -> Tuple[Optional[np.ndarray], float]:
    """
    Used to approximate distractor with a sphere for easier calculations.

    :param env: Environment with a distractor.
    :return: Centre and radius of the distractor represented as a sphere if a distractor exists.
    """
    distractors = env.get_distractors()
    if not distractors:
        return None, 0.0

    d = distractors[0]
    uid = d.get_uid()

    aabb_min, aabb_max = env.p.getAABB(uid)
    aabb_min = np.array(aabb_min, dtype=float)
    aabb_max = np.array(aabb_max, dtype=float)

    center = 0.5 * (aabb_min + aabb_max)
    radius = 0.5 * float(np.linalg.norm(aabb_max - aabb_min))

    return center, radius