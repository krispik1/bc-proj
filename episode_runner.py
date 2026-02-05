from typing import List

import numpy as np

from dataset_types import Episode, Transition, PlannerMode, PlannerPolicy
from helpers.geometry_helper import dist
from wrapper import get_action_vector, GymWrapper


class EpisodeRunner:

    def __init__(
            self,
            env: GymWrapper,
            episode_horizon: int = 50,
            start_pos: np.ndarray = None,
            goal_pos: np.ndarray = None,
    ) -> None:
        """
        Episode generator that simulates planned trajectories from which we get observations and actions.

        :param env: Gym environment with added functionalities.
        :param episode_horizon: Number of observations an episode should be represented by.
        :param start_pos: Starting position of the robot.
        :param goal_pos: Goal position of the robot.
        """
        self.env = env

        # Remember start position and goal position in joint angles
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        self.episode_horizon = episode_horizon

    def _downsample_episode(
            self,
            episode: Episode,
    ) -> Episode:
        """
        Used to keep number of states and actions consistent across all episodes if possible.

        :param episode: Description of the episode.
        :return: New episode with desired number of states and actions.
        """
        transitions = episode.transitions
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
            s_t = transitions[i].state_t
            s_t1 = transitions[j].state_t1
            step_collision = any(tr.step_collision for tr in episode.transitions[i:j+1])

            new_transitions.append(
                Transition(
                    state_t=s_t,
                    action=get_action_vector(s_t, s_t1),
                    state_t1=s_t1,
                    step_collision=step_collision
                )
            )

        episode.transitions = new_transitions
        return episode

    def run_planned_episode(
        self,
        trajectory: List[np.ndarray],
        planner_policy: PlannerPolicy,
        planner_mode: PlannerMode,
        max_steps_per_segment: int = 50
    ) -> Episode:
        """
        Used to run an episode where the robot follows a planned trajectory in env. The trajectory is planned with
        choosing either direct approach [origin, goal], detour to avoid occlusion [origin, waypoint, goal], or impact
        with distractor [origin, impact] depending on PlannerMode.

        Episode runs in many steps which are then reduced by downsampling the episode transitions for consistent number
        of transitions.

        As it is not guaranteed the trajectory meant for avoidance/collision will actually avoid/collide with the occlusion,
        we use ground-truth approach of checking whether the robot has any contact points with the distractor by querying
        the physics engine used in simulation.

        :param trajectory: Planned trajectory given in joints angles.
        :param planner_policy: Name of the planner that produced the provided trajectory.
        :param planner_mode: Policy of the planner that produced the provided trajectory.
        :param max_steps_per_segment: Upper boundary for number of steps used to reach a waypoint.
        :return: Run's description given as transitions, information about success of the trajectory (avoided/collided)
         and mode of the episode.
        """
        self.env.soft_reset_robot_only()

        # List for observations and collision flag
        episode_transitions: List[Transition] = []
        episode_collision = False

        # Follow trajectory using IK and waypoints and check if there is any collision
        for waypoint in trajectory:

            for i in range(max_steps_per_segment):
                # Get joint angles and turn on magnet if in goal position
                if dist(self.env.get_ef_position(), self.env.get_goal_position()) < 5e-3:
                    self.env.robot.set_magnetization(1)
                else:
                    self.env.robot.set_magnetization(0)

                s_t = self.env.get_state_vector()

                self.env.step(waypoint)
                s_t1 = self.env.get_state_vector()
                # Ground-truth collision
                step_collision = self.env.check_robot_distractor_collision()

                episode_transitions.append(
                    Transition(
                        state_t=s_t,
                        action=get_action_vector(s_t, s_t1),
                        state_t1=s_t1,
                        step_collision=step_collision
                    )
                )

                # If collision, end trajectory (collisions may happen even in "avoid" mode)
                if step_collision:
                    episode_collision = True

                    if not planner_policy == PlannerPolicy.BABBLING:
                        break

                # End effector basically reached the waypoint
                if planner_mode != PlannerMode.COLLIDE:
                    if dist(self.env.robot.calculate_accurate_IK(self.env.get_ef_position()), waypoint) < 1e-6:
                        break

            if episode_collision and not planner_policy == PlannerPolicy.BABBLING:
                break

        # Episode represented by transitions, collision flag, successfully magnetized object flag, chosen planner
        # and mode (flag can be true even in "avoid")
        episode = Episode(
            transitions=episode_transitions,
            episode_collision=episode_collision,
            planner_policy=planner_policy,
            planner_mode=planner_mode,
            success=len(self.env.robot.magnetized_objects) > 0
        )

        # Reduce number of samples for consistent number of states and actions
        episode = self._downsample_episode(
            episode
        )
        return episode