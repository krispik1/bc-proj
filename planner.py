import random
from typing import List

import numpy as np

from dataset_types import PlannerMode, PlannerPolicy
from helpers.geometry_helper import find_ee_position_for_link_collision
from policies.babbling_policy import MotorBabblingGenerator
from policies.rrt_policy import RRTTrajectoryGenerator
from policies.waypoint_policy import WaypointTrajectoryGenerator
from wrapper import GymWrapper

class Planner:
    def __init__(
            self,
            env: GymWrapper,
            target_collision_ratio: float = 0.5,
            start_pos: np.ndarray = None,
            goal_pos: np.ndarray = None,
    ) -> None:
        """
        Planner that utilizes various policies to generate trajectories which are followed in episodes from
        which we get observations and actions.

        :param env: Gym environment with added functionalities.
        :param target_collision_ratio: Desired ratio of colliding trajectories to avoiding trajectories in the dataset.
        :param start_pos: Starting position of the robot.
        :param goal_pos: Goal position of the robot.
        """
        self.env = env
        self.planner_mode = PlannerMode.DIRECT

        # Initialize trajectory generators
        self.babbling_gen = MotorBabblingGenerator()
        self.waypoint_gen = WaypointTrajectoryGenerator(env)
        self.rtt_gen = RRTTrajectoryGenerator(env)

        # Remember start position and goal position in joint angles
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        self.target_collision_ratio = target_collision_ratio

    def _choose_mode(
            self,
            current_ratio: float
    ) -> None:
        """
        Used to choose a mode of the planner, whether it plans a colliding, avoiding and direct trajectory.
        """
        # Always try direct approach first
        if current_ratio == 0:
            self.planner_mode = PlannerMode.DIRECT

        # Based on target vs current ratio, bias the modes but allow both
        if current_ratio < self.target_collision_ratio:
            self.planner_mode = random.choices(
                [PlannerMode.COLLIDE, PlannerMode.AVOID],
                weights=[0.7, 0.3],
                k=1
            )[0]
        else:
            self.planner_mode = random.choices(
                [PlannerMode.AVOID, PlannerMode.COLLIDE],
                weights=[0.7, 0.3],
                k=1
            )[0]



    def plan(
            self,
            planner_policy: PlannerPolicy,
            current_ratio: float = 0.0,
            planner_mode: PlannerMode = None
    ) -> List[np.ndarray]:
        """
        Generates trajectory based on chosen planner algorithm and its mode given in joint angles:
            - babbling - random babbling in the environment represented by joint angles [q_1, q_2, ..., q_n] sampled from
            intervals given by q_max, q_min
            - waypoint - [detour_waypoint, goal] if "avoid", [impact_point] if "collide",
            [] if failed to find trajectory,
            - rrt - [q_start, q_1, ..., q_goal] where q_goal = q_goal_obj_pos if "avoid", q_goal = q_collision if "collide"
            and collision happens before reaching q_goal in configuration q_collision, and q_goal = q_d_colliding if "collide",
            or None if no trajectory was found,
            - rrt + waypoint - combines waypoint calculation from which we get waypoints representing trajectory T and
            through pairwise iteration (q_start, q_goal) from T, we plan trajectory [q_start, q_1, ..., q_goal] and join
            them.

        :param planner_mode: Possibility to choose the mode of the planner for testing.
        :param current_ratio: Current ratio of colliding and avoiding trajectories in the dataset.
        :param planner_policy: Policy of the planner.
        :return: Joints angle trajectory.
        """
        self.env.soft_reset_robot_only()

        if planner_mode is None:
            self._choose_mode(current_ratio)
        else:
            self.planner_mode = planner_mode

        if planner_policy == PlannerPolicy.BABBLING:
            return self.babbling_gen.plan()
        elif planner_policy == PlannerPolicy.WAYPOINT:
            return self.waypoint_gen.plan(self.planner_mode)
        elif planner_policy == PlannerPolicy.RRT:
            # Goal position for "avoid" is the position of the goal object, for "collide" it is the centre of the distractor
            goal = self.goal_pos
            if self.planner_mode == PlannerMode.COLLIDE:
                impact = find_ee_position_for_link_collision(self.env)
                goal = self.env.robot.calculate_accurate_IK(impact)
            return self.rtt_gen.plan(
                self.start_pos,
                goal,
                self.planner_mode
            )
        else:
            # For rrt + waypoint, trajectory is planned by planning subtrajectories with rrt algorithm from
            # waypoints provided by WaypointPlanner
            waypoints = [self.start_pos] + self.waypoint_gen.plan(self.planner_mode)
            traj = []
            self.env.soft_reset_robot_only()

            for i in range(len(waypoints) - 1):
                traj += self.rtt_gen.plan(
                    waypoints[i],
                    waypoints[i + 1],
                    self.planner_mode
                )

            return traj