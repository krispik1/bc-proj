import numpy as np
from typing import List, Tuple

from dataset_types import PlannerMode
from wrapper import GymWrapper
from helpers.geometry_helper import dist, get_distractor_centre, get_distractor_sphere_approximation_radius


class WaypointTrajectoryGenerator:

    def __init__(
            self,
            env: GymWrapper
    ):
        """
         Generator used to generate dataset of trajectories in environment with distractors through waypoints.

         Unlike other generators, WaypointTrajectoryGenerator works with positions of waypoints but returns trajectory
         in joints angles for waypoints.

        :param env: Gym environment with added functionalities.
        """
        self.env = env

        # Memorize positions for planning
        self.start_pos = self.env.get_ef_position()
        self.goal_pos = self.env.get_goal_position()

        # Approximate the distractor by a sphere using bounding boxes
        aabb_min, aabb_max = self.env.p.getAABB(self.env.get_distractors()[0].get_uid())
        aabb_min = np.array(aabb_min, dtype=float)
        aabb_max = np.array(aabb_max, dtype=float)
        self.distractor_centre = get_distractor_centre(aabb_min, aabb_max)
        self.distractor_radius = get_distractor_sphere_approximation_radius(aabb_min, aabb_max)

    # In md

    def _compute_detour_waypoint(
            self,
            base_clearance: float = 0.05,
            scale_radius: Tuple[float, float] = (0.8, 1.5),
            t_sigma: float = 0.15,
    ) -> np.ndarray:
        # Calculate tangent
        direct_path = self.goal_pos - self.start_pos
        direct_path_norm = float(np.linalg.norm(direct_path))
        if direct_path_norm < 1e-9:
            # Start and goal pos are almost the same
            return self.goal_pos.copy()
        t_hat = direct_path / (direct_path_norm + 1e-9)

        # Find parameter of line going through the closest point on trajectory and distractor centre
        direct_path_norm2 = float(np.dot(direct_path, direct_path)) + 1e-9
        t_closest = float(np.dot(self.distractor_centre - self.start_pos, direct_path)) / direct_path_norm2
        t_closest = np.clip(t_closest, 0.0, 1.0)

        # Randomly move it and find the closest point to distractor on trajectory
        t_anchor = np.random.normal(loc=t_closest, scale=t_sigma)
        t_anchor = np.clip(t_anchor, 0.05, 0.95)
        closest_point_to_occ = self.start_pos + t_anchor * direct_path

        # Find orthogonal plane
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(axis, t_hat)) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)

        u = np.cross(t_hat, axis)
        u_norm = float(np.linalg.norm(u))
        if u_norm < 1e-9:
            # u is unusable
            return self.goal_pos.copy()
        u /= u_norm + 1e-9

        w = np.cross(t_hat, u)
        w /= float(np.linalg.norm(w)) + 1e-9

        clearance = self.distractor_radius + base_clearance

        # Sample direction and length of the push
        theta = np.random.uniform(0, 2 * np.pi)
        scale = np.random.uniform(scale_radius[0], scale_radius[1])
        r = clearance * scale

        dir_offset = np.cos(theta) * u + np.sin(theta) * w
        waypoint = closest_point_to_occ + r * dir_offset

        return waypoint

    def _sample_collision_target_on_distractor(
            self,
            max_tries: int = 10000,
            phi_max: float = 60
    ) -> List[np.ndarray]:

        # Find tangent of path to the centre of distractor
        d = self.distractor_centre - self.start_pos
        d_norm = float(np.linalg.norm(d))

        # Prevent division by zero
        if d_norm < 1e-9:
            return [self.distractor_centre]
        d /= (d_norm + 1e-9)

        cos_max = np.cos(np.deg2rad(phi_max))

        for _ in range(max_tries):
            # Randomly sample unit vector and check if it is in cone directed from start to distractor
            v = np.random.normal(size=3)
            v_norm = float(np.linalg.norm(v))

            # Prevent division by zero
            if v_norm < 1e-9:
                continue
            v /= (v_norm + 1e-9)

            if np.dot(v, d) < cos_max:
                continue

            # Randomize the impact point and check if it is valid
            r = np.random.uniform(0.1 * self.distractor_radius, 0.5 * self.distractor_radius)
            return [self.distractor_centre + r * v]

        return [self.distractor_centre]

    def _plan_direct(
            self
    ) -> List[np.ndarray]:
        """
        Used to plan a direct trajectory.

        :return: Goal position.
        """
        return [self.goal_pos]

    def _plan_avoidance(
            self
    ) -> List[np.ndarray]:
        """
        Used to plan trajectories that should avoid collision.

        :return: Trajectory given in points of detour waypoint and goal point if there is enough distance between them.
        Otherwise, only goal point is given.
        """
        waypoint = self._compute_detour_waypoint()
        if not dist(waypoint, self.goal_pos):
            return [self.goal_pos]

        return [waypoint, self.goal_pos]

    def _plan_collide(
            self
    ) -> List[np.ndarray]:
        """
        Used to plan trajectories that should cause collision.

        :return: Impact point for the trajectory.
        """
        return self._sample_collision_target_on_distractor()

    def _valid_traj(
            self,
            traj: List[np.ndarray],
            mode: PlannerMode
    ) -> bool:
        """
        Validates trajectory.

        :param traj: Trajectory in joint angles.
        :param mode: Mode of the planner.
        :return: True only if collision in "collide" or no collision in "avoid".
        """
        for q in traj:
            # Move end effector and check if there is a collision without simulating it
            self.env.set_robot_configuration_kinematic(q)
            self.env.p.stepSimulation()
            collision_flag = self.env.check_robot_distractor_collision()

            if collision_flag:
                return mode == PlannerMode.COLLIDE

        return not mode == PlannerMode.COLLIDE

    def plan(
            self,
            mode: PlannerMode,
            max_tries: int = 100
    ) -> List[np.ndarray]:
        """
        Plans a trajectory based on the selected planner mode using a detour waypoint/impact point.

        :param max_tries: Upper bound for repeated tries to construct a trajectory.
        :param mode: Type of planner to use.
        :return: Lists of joint angles for each waypoint from planned trajectory.
        """
        for _ in range(max_tries):
            # Plan traj based on mode
            if mode == PlannerMode.AVOID:
                waypoints = self._plan_avoidance()
            elif mode == PlannerMode.COLLIDE:
                waypoints = self._plan_collide()
            else:
                waypoints = self._plan_direct()

            # Calculate for [x, y, z] waypoint corresponding joint angles with IK
            traj = [self.env.robot.calculate_accurate_IK(waypoint) for waypoint in waypoints]

            # Validate traj before simulation
            if self._valid_traj(traj, mode=mode):
                return traj

        return []