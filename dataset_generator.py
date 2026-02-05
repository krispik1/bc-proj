from typing import List

from dataset_types import PlannerPolicy, Episode
from episode_runner import EpisodeRunner
from planner import Planner
from wrapper import GymWrapper

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

        start_pos = env.robot.calculate_accurate_IK(env.get_ef_position())
        goal_pos = env.robot.calculate_accurate_IK(env.get_goal_position())

        # Initialize planner and episode generator
        self.planner = Planner(env, target_collision_ratio, start_pos, goal_pos)
        self.episode_gen = EpisodeRunner(env, episode_horizon, start_pos, goal_pos)

        # List of planners and number of episodes for each
        self.planners = [PlannerPolicy.BABBLING, PlannerPolicy.WAYPOINT, PlannerPolicy.RRT, PlannerPolicy.RRT_WAYPOINT]
        self.n_episodes_for_planners = n_episodes_for_planners

        # Bookkeeping for balancing the ratio
        self._n_collision = 0
        self._n_non_collision = 0
        self._episode_index = 0

    def collect_data(
        self
    ) -> List[Episode]:
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
        and collision flag (True only if collision), information about the planner and success of the robot.
        """

        episodes = []

        for planner_index in range(len(PlannerPolicy)):
            for _ in range(self.n_episodes_for_planners[planner_index]):
                current_ratio = 0
                if self._episode_index != 0:
                    current_ratio = self._n_collision / self._n_non_collision

                trajectory = self.planner.plan(
                    planner_policy=PlannerPolicy(planner_index),
                    current_ratio=current_ratio
                )

                episode = self.episode_gen.run_planned_episode(
                    trajectory=trajectory,
                    planner_policy=PlannerPolicy(planner_index),
                    planner_mode=self.planner.planner_mode,
                )

                episodes.append(episode)

                # Bookkeep for choosing modes
                if episode.episode_collision:
                    self._n_collision += 1
                else:
                    self._n_non_collision += 1
                self._episode_index += 1


            self._n_collision = 0
            self._n_non_collision = 0
            self._episode_index = 0

        return episodes