import numpy as np
from typing import List

class MotorBabblingGenerator:

    def __init__(
            self,
            q_min: np.ndarray = np.array([-2.967, -1.833, -2.967, -3.142, -2.967, -0.087, -2.967]),
            q_max: np.ndarray = np.array([2.967, 1.833, 2.967, 0.0, 2.967, 3.822, 2.967])
    ):
        """
        Generator used to generate dataset of motor babbling in environment with distractors.

        :param q_min: Array of minimum values of each joint angle from which an action is sampled.
        :param q_max: Array of maximum values of each joint angle from which an action is sampled.
        """

        # Values of joint angles for random normal sampling of an action
        self.q_min = q_min
        self.q_max = q_max

    def _sample_action(
            self
    ) -> np.ndarray:
        """
        Used to sample a random list of joint angles from uniform distribution.

        :return: Array of joint angles the robot should reach representing action.
        """
        return np.random.uniform(self.q_min, self.q_max)

    def plan(
            self,
            n_steps: int = 10
    ) -> List[np.ndarray]:
        """
        Plans a babbling trajectory with given number of steps. Aims to create random movement exploring the environment.

        :param n_steps: Number of steps the robot takes.
        :return: List of joint angles for each step.
        """

        return [self._sample_action() for _ in range(n_steps)]