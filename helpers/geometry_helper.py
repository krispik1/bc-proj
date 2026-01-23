import numpy as np
from typing import Tuple, Optional

from wrapper import GymWrapper

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

def sample_surface_point_ray(
        env: GymWrapper,
        n_rays: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Using ray test provided by the physics engine, we sample a point from the true surface of the distractor. For greater
    randomness in sampled rays, we randomly choose one of the faces of axis aligned bounding box for origin point of the
    ray while uniformly sampling other coordinates.

    :param env: Gym environment with modified functionality.
    :param n_rays: Number of rays to use for ray tests.
    :return: Sampled position and normal on true surface of the distractor.
    """
    # Random number generator for choosing faces
    rng = np.random.default_rng()

    # Find centre of aabb
    aabb_min, aabb_max = env.p.getAABB(env.get_distractors()[0].get_uid())
    aabb_min = np.array(aabb_min, dtype=float)
    aabb_max = np.array(aabb_max, dtype=float)
    center = get_distractor_centre(aabb_min, aabb_max)

    # Sample rays for ray test batch - ray starts from surface of aabb and ends in centre of aabb/distractor
    rays_from, rays_to = [], []
    for _ in range(n_rays):
        # Pick randomly one of six faces of rectangular block that is aabb
        face = rng.integers(0, 6)

        x = np.random.uniform(aabb_min[0], aabb_max[0])
        y = np.random.uniform(aabb_min[1], aabb_max[1])
        z = np.random.uniform(aabb_min[2], aabb_max[2])

        if face == 0:
            x = aabb_max[0]
        elif face == 1:
            x = aabb_min[0]
        elif face == 2:
            y = aabb_max[1]
        elif face == 3:
            y = aabb_min[1]
        elif face == 4:
            z = aabb_max[2]
        else:
            z = aabb_min[2]

        start = np.array([x, y, z], dtype=float)
        end = center
        rays_from.append(start.tolist())
        rays_to.append(end.tolist())

    results = env.p.rayTestBatch(rays_from, rays_to)

    # Test the results and return position and norm of first suitable ray test result
    for hit_uid, hit_link, hit_frac, hit_pos, hit_norm in results:
        # Ray must hit the distractor
        if hit_uid != env.get_distractors()[0].get_uid() or hit_frac < 0:
            continue

        p_world = np.array(hit_pos, dtype=float)
        n_world = np.array(hit_norm, dtype=float)
        n_world /= max(np.linalg.norm(n_world), 1e-6)
        return p_world, n_world

    # Only if no ray hit the distractor
    raise RuntimeError

def find_ee_position_for_link_collision(
        env: GymWrapper,
        approach_start: float = 0.2,
        approach_end: float = -0.02,
        steps: int = 30
) -> np.ndarray:
    """
    As the end effector has no real colliding body, this function is used to find end effector position, using sampled
    point on the surface of the distractor, resulting in collision caused by any link of the robot that is able to collide.

    End effector slowly approaches the sampled point from outside the distractor until a collision occurs or the
    end effector is not located some distance from the sampled point inside the distractor.

    :param env: Gym environment with modified functionality.
    :param approach_start: Starting distance away from the sampled point outside the distractor.
    :param approach_end: Maximum end distance inside the distractor from the sampled point.
    :param steps: Number of steps representing number of nudges.
    :return: End effector position that causes collision.
    """
    # Sample point on surface
    hit_p, hit_n = sample_surface_point_ray(env)

    # Values that move the end effector from start to end through the sampled point, from outside to inside the distractor
    s_values = np.linspace(approach_start, approach_end, steps)

    # Repeatedly nudge the end effector and test for collision
    for s in s_values:
        ee_pos = hit_p + s * hit_n
        q = env.robot.calculate_accurate_IK(ee_pos)

        if q is None:
            continue

        env.set_robot_configuration_kinematic(q)

        if env.check_robot_distractor_collision():
            return ee_pos

    # If no end effector position was found
    aabb_min, aabb_max = env.p.getAABB(env.get_distractors()[0].get_uid())
    aabb_min = np.array(aabb_min, dtype=float)
    aabb_max = np.array(aabb_max, dtype=float)
    return get_distractor_centre(aabb_min, aabb_max)

def get_distractor_centre(
        aabb_min: np.ndarray,
        aabb_max: np.ndarray
) -> Optional[np.ndarray]:
    """
    Used to calculate the centre point of the distractor using aabb.

    :param aabb_min: Minimum coordinates of the axis aligned bounding box.
    :param aabb_max: Maximum coordinates of the axis aligned bounding box.
    :return: Centre of the distractor.
    """
    return 0.5 * (aabb_min + aabb_max)

def get_distractor_sphere_approximation_radius(
        aabb_min: np.ndarray,
        aabb_max: np.ndarray
) -> Optional[np.ndarray]:
    """
    Used to calculate the radius of the sphere approximating the distractor using aabb.

    :param aabb_min: Minimum coordinates of the axis aligned bounding box.
    :param aabb_max: Maximum coordinates of the axis aligned bounding box.
    :return: Radius of the sphere approximating the distractor.
    """
    extents = aabb_max - aabb_min
    return 0.5 * np.linalg.norm(extents)