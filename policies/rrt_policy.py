from typing import Tuple, List

import numpy as np

from dataset_types import PlannerMode
from wrapper import GymWrapper
from helpers.geometry_helper import dist


def _is_jagged_triangle(
        q_0: np.ndarray,
        q_1: np.ndarray,
        q_2: np.ndarray,
        angle_degree: float = 60.0
) -> bool:
    """
    Tests whether the joints configurations create a segment of trajectory that looks jagged.

    :param q_0: First joints configuration represented by angles.
    :param q_1: Second joints configuration represented by angles.
    :param q_2: Third joints configuration represented by angles.
    :param angle_degree: Maximum turn angle in degrees which the robot's end effector is allowed to make for smoother
    looking trajectory.
    :return: True only if the triangle formed by given joints configuration is too sharp.
    """
    v1 = q_1 - q_0
    v2 = q_2 - q_0
    n1 = dist(q_0, q_1)
    n2 = dist(q_1, q_2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0 >= angle_degree

    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    ang = float(np.degrees(np.arccos(c)))

    return ang >= angle_degree


class RRTTrajectoryGenerator:

    def __init__(
            self,
            env: GymWrapper,
            step_size: float = 0.1,
            edge_substeps: int = 20,
            goal_bias: float = 0.2
    ):
        """
        Generator used to generate dataset of trajectories in environment with distractors through Rapidly exploring
        Random Trees.

        :param env: Gym environment with added functionalities.
        :param step_size: Maximum length of movement during a step representing length of an edge.
        :param edge_substeps: Number of interpolation steps per edge.
        :param goal_bias: Bias in sampling steering trajectory to the goal object.
        """
        self.env = env

        self.step_size = step_size
        self.edge_substeps = edge_substeps

        self.goal_bias = goal_bias

        # Joint limits provide sample space for configurations
        self.lows, self.highs = self.env.robot.joints_limits

    def _sample_joints_angles(
            self
    ) -> np.ndarray:
        """
        Used to sample joint configuration from the joint limits of the robot.

        :return: Configuration of joints represented by angles.
        """
        return np.array(
            [np.random.uniform(low, high) for low, high in zip(self.lows, self.highs)],
            dtype=float,
        )

    def _steer(
            self,
            q_from: np.ndarray,
            q_to: np.ndarray,
    ) -> np.ndarray:
        """
        Used to steer direction of trajectory towards chosen configuration.

        :param q_from: Current joint configuration.
        :param q_to: Joint configuration we want to steer towards.
        :return: Joint configuration respecting max step size that steers the robot towards desired configuration.
        """
        direction = q_to - q_from
        d = dist(q_from, q_to)

        if not d:
            return q_from

        return q_from + direction / d * min(self.step_size, d)

    def _check_validity(
            self,
            q: np.ndarray,
            allow_hit: bool
    ) -> Tuple[bool, bool]:
        """
        Checks whether the movement performed by the robot is allowed.

        When collision is present, detected by the physics engine, the movement is only valid if impact with distractor is
        allowed as it is allowed in collision mode of planner.

        :param q: Joint configuration for which we check validity.
        :param allow_hit: True during collision planning which allows impacts.
        :return: Is movement valid and did collision happen.
        """
        self.env.set_robot_configuration_kinematic(q)
        collision_flag = self.env.check_robot_distractor_collision()
        self.env.p.stepSimulation()

        return (allow_hit or not collision_flag), collision_flag

    def _edge_valid(
            self,
            q_from: np.ndarray,
            q_to: np.ndarray,
            mode: PlannerMode
    ) -> Tuple[bool, bool]:
        """
        Checks whether the edge is valid by checking movements for each substep during the edge.

        Edge is valid if each substep is a valid movement checked by _check_config().

        :param q_from: Current joint configuration.
        :param q_to: Next desired configuration.
        :param mode: Mode of planner.
        :return: Is edge valid and did collision happen.
        """
        collision_flag = False

        # Check for each substep its validity
        for i in range(self.edge_substeps + 1):
            t = i / float(self.edge_substeps)
            q = q_from + t * (q_to - q_from)

            if mode == PlannerMode.AVOID:
                valid, collision = self._check_validity(q, allow_hit=False)
            else:
                valid, collision = self._check_validity(q, allow_hit=True)

            if collision:
                collision_flag = True

            if not valid:
                return False, collision_flag

        return True, collision_flag

    def _rtt(
            self,
            q_start: np.ndarray,
            q_goal: np.ndarray,
            mode: PlannerMode,
            max_iterations: int = 1000,
    ) -> List[np.ndarray]:
        """
        Uses RRT algorithm with tree data structure to find trajectory from starting configuration to goal configuration.
        Sampling is biased to steer the trajectory to the goal.

        :param q_start: Starting joint configuration.
        :param q_goal: Goal joint configuration.
        :param mode: Mode of planner.
        :param max_iterations: Upper bound on number of tree nodes.
        :return: List of joint angles representing trajectory.
        """
        # Initialize tree structure for RRT
        nodes: List[np.ndarray] = [q_start]
        parents: List[int] = [-1]

        for i in range(max_iterations):
            # Bias the sampling process
            if np.random.random() < self.goal_bias:
                q_sample = q_goal
            else:
                q_sample = self._sample_joints_angles()

            # Find nearest neighbour
            distances = [dist(q, q_sample) for q in nodes]
            q_near_idx = int(np.argmin(distances))
            q_near = nodes[q_near_idx]

            # Steer towards the sample
            q_new = self._steer(q_near, q_sample)

            # Check if new edge is valid then add new configuration to RRT
            edge_valid, collision_flag = self._edge_valid(q_near, q_new, mode)
            if not edge_valid:
                continue

            nodes.append(q_new)
            parents.append(q_near_idx)
            q_new_idx = len(nodes) - 1

            # If collide mode and collision occurred, new configuration is final
            if mode == PlannerMode.COLLIDE and collision_flag:
                traj: List[np.ndarray] = [q_new.copy()]
            # Else if we are close to goal configuration, connect new and goal, and goal is final
            elif dist(q_new, q_goal) <= self.step_size:
                edge_valid_goal, _ = self._edge_valid(q_new, q_goal, mode)
                if not edge_valid_goal:
                    continue

                traj: List[np.ndarray] = [q_goal.copy(), q_new.copy()]
            else:
                continue

            # Traverse the tree to find trajectory
            parent = parents[q_new_idx]
            while parent != -1:
                traj.append(nodes[parent])
                parent = parents[parent]
            traj.reverse()
            return traj

        return []

    def _shortcut_valid(
            self,
            q_0: np.ndarray,
            q_1: np.ndarray,
            mode: PlannerMode,
            max_shortcut_length: float = 0.25,
    ) -> bool:
        """
        Tests if new shortcut created during trajectory is valid.

        :param q_0: Starting joints configuration.
        :param q_1: End joints configuration.
        :param mode: Mode of the planner.
        :param max_shortcut_length: Maximum length of path.
        :return: True only if the new edge representing the shortcut is valid.
        """
        if dist(q_0, q_1) > max_shortcut_length:
            return False

        ok, _ = self._edge_valid(q_0, q_1, mode)

        return ok

    def _smooth_trajectory_through_removal(
            self,
            trajectory: List[np.ndarray],
            mode: PlannerMode,
            angle_degree: float = 60.0,
            max_shortcut_length: float = 0.25,
            min_nodes: int = 10,
            max_passes: int = 10,
    ) -> List[np.ndarray]:
        """
        Used to smooth a trajectory planned by RRT algorithm to make them more humanlike.

        This approach iteratively tests three consecutive joints configurations and tests whether the middle joints
        configuration makes the segment look too jagged. If it does, it is removed from the trajectory following
        restrictions to keep some randomness to the trajectories that are generated.

        :param trajectory: RRT trajectory.
        :param mode: Mode of the planner.
        :param angle_degree: Maximum turn angle in degrees that the robot takes for smoother look of the trajectory.
        :param max_shortcut_length: Maximum length of shortcut which would be a result of removing configuration.
        :param min_nodes: Minimum number of configurations that represent the trajectory.
        :param max_passes: Maximum iterations through the trajectory.
        :return: Smoothed trajectory by removal of joints configurations.
        """
        # Return if there are no nodes to be removed.
        if len(trajectory) <= min_nodes:
            return trajectory

        for _ in range(max_passes):
            if len(trajectory) <= min_nodes:
                break

            changed = False
            i = 0

            # Test the three consecutive configurations
            while i <= len(trajectory) - 3 and len(trajectory) > min_nodes:
                q0, q1, q2 = trajectory[i], trajectory[i + 1], trajectory[i + 2]

                # q1 must create sharp turns and new path q0->q2 must be valid for q1 to be removed
                if _is_jagged_triangle(q0, q1, q2, angle_degree):
                    if self._shortcut_valid(q0, q2, mode, max_shortcut_length):
                        trajectory.pop(i + 1)
                        changed = True

                        i = max(i - 1, 0)
                        continue

                i += 1

            # If iteration produced no change, so will the next one - end loop
            if not changed:
                break

        return trajectory

    def _smooth_trajectory_through_elastic_band(
            self,
            trajectory: List[np.ndarray],
            mode: PlannerMode,
            max_passes: int = 30,
            lam: float = 0.35,
            max_step: float = 0.05,
    )-> List[np.ndarray]:
        """
        Used to smooth a trajectory planned by RRT algorithm to make them more humanlike.

        This approach iteratively takes three consecutive joints configurations and

        :param trajectory: RRT trajectory.
        :param mode: Mode of the planner.
        :param max_passes: Maximum iterations through the trajectory.
        :param lam: Control input for aggressiveness of smoothing (small = gentle smoothing, large = fast smoothing).
        :param max_step: Control input for magnitude of jump when smoothing to prevent jumps to unreachable areas.
        :return: Smoothed trajectory by movement of joints configurations to more natural configuration
        like an elastic band (Laplacian smoothing).
        """
        # No possible change if trajectory consists of only two configurations
        if len(trajectory) <= 2:
            return trajectory

        # Get joints limits to not produce invalid configurations by pushes
        lows, highs = self.env.robot.joints_limits

        for _ in range(max_passes):
            changed = False

            # Smooth a segment of three consecutive configurations
            for i in range(1, len(trajectory) - 1):
                q_prev = trajectory[i - 1]
                q = trajectory[i]
                q_next = trajectory[i + 1]

                # Choose target position based on neighbouring configurations
                q_target = 0.5 * (q_prev + q_next)

                # Calculate change to the middle configuration
                # Control speed of smoothing
                dq = lam * (q_target - q)
                n = float(np.linalg.norm(dq))
                # Control magnitude of change
                if n > max_step:
                    dq = dq / n * max_step

                # Change is too small
                if float(np.linalg.norm(dq)) < 1e-6:
                    continue

                # Smoothing operation - find new configuration respecting the joints limits
                q_new = np.clip(q + dq, lows, highs)

                # Test if new configuration creates valid paths
                ok1, _ = self._edge_valid(q_prev, q_new, mode)
                if not ok1:
                    continue

                ok2, _ = self._edge_valid(q_new, q_next, mode)
                if not ok2:
                    continue

                # Replace old configuration with the new one if it satisfies conditions
                trajectory[i] = q_new
                changed = True

            # If iteration produced no change, so will the next one - end loop
            if not changed:
                break

        return trajectory

    def plan(
            self,
            q_start: np.ndarray,
            q_goal: np.ndarray,
            mode: PlannerMode,
            n_restarts: int = 5
    ) -> List[np.ndarray]:
        """
        Plans a trajectory based on the selected planner mode using a Rapidly-exploring Random Tree.

        :param q_start: Starting joint configuration.
        :param q_goal: Desired joint configuration that the robot should reach.
        :param mode: Mode of planner.
        :param n_restarts: Number of restarts to find a valid trajectory.
        :return: Valid trajectory that avoids/collides with collision based on planner mode if one was found.
        """
        # If starting and goal position are practically identical
        if not dist(q_start, q_goal):
            return [q_start.copy()]

        # Try to find a trajectory using RRT
        for _ in range(n_restarts):
            traj = self._rtt(q_start, q_goal, mode)
            if traj:
                return traj

        return []