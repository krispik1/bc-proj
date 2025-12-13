from typing import Literal, Tuple, List

import numpy as np

from wrapper import GymWrapper
from helpers.geometry_helper import dist


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

    PlannerMode = Literal["direct", "avoid", "collide"]

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

            if mode == "avoid":
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
            if mode == "collide" and collision_flag:
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