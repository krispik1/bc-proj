from typing import Literal, Dict, Any, List, get_args

import numpy as np

from wrapper import GymWrapper, get_action_vector
from helpers.geometry_helper import get_distractor_sphere, dist
from policies.babbling_policy import MotorBabblingGenerator
from policies.rrt_policy import RRTTrajectoryGenerator
from policies.waypoint_policy import WaypointTrajectoryGenerator

# Literals for planners and their modes
PlannerMode = Literal["direct", "avoid", "collide"]
Planner = Literal["babbling", "waypoint", "rrt", "rrt_waypoint"]

class DatasetGenerator:

    def __init__(
            self,
            env: GymWrapper,
            episode_horizon: int = 50,
            target_collision_ratio: float = 0.5,
            n_episodes_for_planners: List[int]=None
    ):
        """
        Dataset generator that utilizes various planners to generate trajectories which are followed in episodes from
        which we get observations and actions.

        :param env: Gym environment with added functionalities.
        :param episode_horizon: Number of observations an episode should be represented by.
        :param target_collision_ratio: Desired ratio of colliding trajectories to avoiding trajectories in the dataset.
        :param n_episodes_for_planners: Number of episodes for each planner.
        """
        if n_episodes_for_planners is None:
            n_episodes_for_planners = [20, 20, 20, 20]

        self.env = env
        # Remember start position and goal position in joint angles
        self.start_pos = self.env.robot.calculate_accurate_IK(env.get_ef_position())
        self.goal_pos = self.env.robot.calculate_accurate_IK(env.get_goal_position())

        self.episode_horizon = episode_horizon
        self.target_collision_ratio = target_collision_ratio

        # Bookkeeping for balancing the ratio
        self._n_collision = 0
        self._n_non_collision = 0
        self._episode_index = 0

        # Initialize trajectory generators
        self.babbling_gen = MotorBabblingGenerator()
        self.waypoint_gen = WaypointTrajectoryGenerator(env)
        self.rtt_gen = RRTTrajectoryGenerator(env)

        # List of planners and number of episodes for each
        self.planners = ["babbling", "waypoint", "rrt", "rrt_waypoint"]
        self.n_episodes_for_planners = n_episodes_for_planners

    def _choose_mode(
            self
    ) -> PlannerMode:
        """
        Used to choose a mode of the planner, whether it plans a colliding, avoiding and direct trajectory.

        :return: Literal representing the mode with possible options of direct, avoid or collide mode.
        """
        # Always try direct approach first
        if self._episode_index == 0:
            return "direct"

        current_ratio = self._n_collision / self._episode_index

        # Based on target vs current ratio, bias the modes but allow both
        if current_ratio < self.target_collision_ratio:
            return np.random.choice(
                ["collide", "avoid"],
                p=[0.7, 0.3]
            )
        else:
            return np.random.choice(
                ["avoid", "collide"],
                p=[0.7, 0.3]
            )

    def _downsample_episode(
            self,
            episode: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Used to keep number of states and actions consistent across all episodes if possible.

        :param episode: Description of the episode.
        :return: New episode with desired number of states and actions.
        """
        transitions = episode["transitions"]
        n_transitions = len(transitions)

        #If no transitions or too little, keep as is
        if n_transitions == 0:
            return episode
        if n_transitions <= self.episode_horizon:
            return episode

        # Choose t+1 transitions evenly spaced through the episode
        idx_states = np.linspace(
            0,
            n_transitions-1,
            num=self.episode_horizon + 1,
            endpoint=True)
        idx_states = np.round(idx_states).astype(int)
        idx_states = np.clip(idx_states, 0, n_transitions-1)

        # Create new transitions by joining old transitions based on indices and create new action vector for each
        new_transitions = []
        for k in range(self.episode_horizon):
            i = idx_states[k]
            j = idx_states[k + 1]
            s_t = transitions[i]["s_t"]
            s_t1 = transitions[j]["s_t1"]

            new_transitions.append(
                {
                    "s_t": s_t,
                    "a_t": get_action_vector(s_t, s_t1),
                    "s_t1": s_t1,
                }
            )

        episode["transitions"] = new_transitions
        return episode

    def _get_trajectory(
            self,
            mode: PlannerMode,
            planner: Planner
    ) -> List[np.ndarray]:
        """
        Generates trajectory based on chosen planner algorithm and its mode given in joint angles:
            - babbling - random babbling in the environment represented by joint angles [q_1, q_2, ..., q_n] sampled from
            intervals given by q_max, q_min
            - waypoint - [detour_waypoint, goal] if "avoid", [impact_point] if "collide",
            [] if failed to find trajectory,
            - rrt - [q_start, q_1, ..., q_goal] where q_goal = q_goal_obj_pos if "avoid", q_goal = q_collision if "collide"
            and collision happens before reaching q_goal in configuration q_collision, and q_goal = q_d_centre if "collide",
            or None if no trajectory was found,
            - rrt + waypoint - combines waypoint calculation from which we get waypoints representing trajectory T and
            through pairwise iteration (q_start, q_goal) from T, we plan trajectory [q_start, q_1, ..., q_goal] and join
            them.

        :param mode: Mode of the planner.
        :param planner: Name of the planner.
        :return: Joints angle trajectory.
        """
        if planner == "babbling":
            return self.babbling_gen.plan()
        elif planner == "waypoint":
            return self.waypoint_gen.plan(mode)
        elif planner == "rrt":
            # Goal position for "avoid" is the position of the goal object, for "collide" it is the centre of the distractor
            goal = self.goal_pos
            if mode == "collide":
                centre, _ = get_distractor_sphere(self.env)
                goal = self.env.robot.calculate_accurate_IK(centre)
            return self.rtt_gen.plan(
                self.start_pos,
                goal,
                mode
            )
        else:
            # For rrt + waypoint, trajectory is planned by planning subtrajectories with rrt algorithm from
            # waypoints provided by WaypointPlanner
            waypoints = [self.start_pos] + self.waypoint_gen.plan(mode)
            traj = []
            self.env.soft_reset_robot_only()

            for i in range(len(waypoints) - 1):
                traj += self.rtt_gen.plan(
                    waypoints[i],
                    waypoints[i + 1],
                    mode
                )

            return traj

    def _run_planned_episode(
        self,
        mode: PlannerMode,
        planner: Planner,
        max_steps_per_segment: int = 50
    ) -> Dict[str, Any]:
        """
        Used to run an episode where the robot follows a planned trajectory in env. The trajectory is planned with
        choosing either direct approach [origin, goal], detour to avoid occlusion [origin, waypoint, goal], or impact
        with distractor [origin, impact] depending on PlannerMode.

        Episode runs in many steps which are then reduced by downsampling the episode transitions for consistent number
        of transitions.

        As it is not guaranteed the trajectory meant for avoidance/collision will actually avoid/collide with the occlusion,
        we use ground-truth approach of checking whether the robot has any contact points with the distractor by querying
        the physics engine used in simulation.

        :param mode: Literal used to decide what type of episode we want to run.
        :param max_steps_per_segment: Upper boundary for number of steps used to reach a waypoint.
        :return: Run's description given as transitions, information about success of the trajectory (avoided/collided)
         and mode of the episode.
        """
        # Reset env but keep initial positions of objects and robot
        self.env.soft_reset_robot_only()

        # Plan trajectory based on mode
        trajectory = self._get_trajectory(mode, planner)
        self.env.soft_reset_robot_only()

        # List for observations and collision flag
        episode_transitions: List[Dict[str, Any]] = []
        episode_collision = False

        # Follow trajectory using IK and waypoints and check if there is any collision
        for waypoint in trajectory:

            # Get joint angles and turn on magnet if in goal position
            if dist(waypoint, self.goal_pos) < 5e-3:
                self.env.robot.set_magnetization(1)
            else:
                self.env.robot.set_magnetization(0)

            for _ in range(max_steps_per_segment):
                s_t = self.env.get_state_vector()

                self.env.step(waypoint)
                s_t1 = self.env.get_state_vector()
                # Ground-truth collision
                step_collision = self.env.check_robot_distractor_collision()

                episode_transitions.append(
                    {
                        "s_t": s_t,
                        "a_t": get_action_vector(s_t, s_t1),
                        "s_t1": s_t1,
                        "step_collision": step_collision
                    }
                )

                # If collision, end trajectory (collisions may happen even in "avoid" mode)
                if step_collision:
                    episode_collision = True

                    if not planner == "babbling":
                        break

                # End effector basically reached the waypoint
                if mode != "collide":
                    if dist(self.env.robot.calculate_accurate_IK(self.env.get_ef_position()), waypoint) < 1e-6:
                        break

            if episode_collision and not planner == "babbling":
                break

        # Episode represented by transitions, collision flag, successfully magnetized object flag, chosen planner
        # and mode (flag can be true even in "avoid")
        episode = {
            "transitions": episode_transitions,
            "episode_collision": episode_collision,
            "mode": mode,
            "planner": planner,
            "success": len(self.env.robot.magnetized_objects) > 0
        }

        # Bookkeep for choosing modes
        if episode_collision:
            self._n_collision += 1
        else:
            self._n_non_collision += 1
        self._episode_index += 1

        # Reduce number of samples for consistent number of states and actions
        episode = self._downsample_episode(
            episode
        )
        return episode

    def collect_data(
        self
    ) -> List[Dict[str, Any]]:
        """
        Runs data collection episodes, each with new trajectory the robot follows. These episodes are divided into
        collision and avoidance episodes, the former containing episodes where trajectory led to a collision with
        a distractor/occlusion, and the latter containing successful trajectories. We try to balance the ratio by
        choosing modes in which the episodes are run.

        Based on the number of episodes given for each planner, the dataset uses multiple planners:
            - Motor babbling - Random exploration of environment,
            - Waypoint - Trajectory given by detour waypoint or impact point,
            - RRT - Trajectory created by Rapidly-exploring Random Trees,
            - Waypoint + RRT - generated waypoint is used for detouring, increases randomization.

        :return: Collected data of episodes represented by list of observations given by transitions (s_t -> a_t -> s_t1)
        and collision flag (True only if collision).
        """

        episodes = []

        for n_episodes, plan in zip(self.n_episodes_for_planners, get_args(Planner)):
            for _ in range(n_episodes):
                mode = self._choose_mode()
                episodes.append(
                    self._run_planned_episode(
                        mode=mode,
                        planner=plan
                    )
                )

            self._n_collision = 0
            self._n_non_collision = 0
            self._episode_index = 0

        return episodes