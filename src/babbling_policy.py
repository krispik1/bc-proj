import numpy as np
from typing import List, Dict, Any

from myGym.myGym.envs.gym_env import GymEnv

class MotorBabblingBasedGenerator:

    def __init__(
            self,
            env: GymEnv,
            mgt_prob: float = 0.5,
            q_min: np.ndarray = np.array([-2.967, -1.833, -2.967, -3.142, -2.967, -0.087, -2.967, 0.0]),
            q_max: np.ndarray = np.array([2.967, 1.833, 2.967, 0.0, 2.967, 3.822, 2.967, 0.0]),
    ):
        self.env = env

        self.q_min = q_min
        self.q_max = q_max
        self.q_mu = (self.q_min + self.q_max) / 2
        self.q_sigma = (self.q_max - self.q_min) / 6

        self.mgt_prob = mgt_prob

    def _sample_action(
            self
    ) -> np.ndarray:
        action = np.random.normal(self.q_mu, self.q_sigma)
        return np.clip(action, self.q_min, self.q_max)

    def _random_toggle_mgt(
            self
    ):
        if np.random.rand() < self.mgt_prob:
            self.env.robot.set_magnetization(1)
        else:
            self.env.robot.set_magnetization(0)

    def _check_robot_distractor_collision(self) -> bool:
        distractors = self.env.task_objects.get("distractor", None)
        if distractors is None:
            return False

        for d in distractors:
            cps = self.env.p.getContactPoints(self.env.robot.get_uid(), d.get_uid())
            if len(cps) > 0:
                return True

        return False


    def episode_zero(
            self
    ) -> Dict[str, Any]:
        s0 = self.env.get_observation()

        return {
            "joint_angles": s0["joints_angles"],
            "ee6D": s0["endeff_6D"],
            "obj6D": s0["obj_6D"],
            "occ": s0["distractor"],
            "mgt": self.env.robot.use_magnet,
        }

    def collect_episode_data(
            self,
            steps: int
    ) -> List[Dict[str, Any]]:
        self.env.reset()
        self.env.robot.set_magnetization(0)

        observations = []

        for t in range(steps):
            s_t = self.env.get_observation()
            s_t_q = np.array(s_t["joints_angles"])
            s_t_mgt = self.env.robot.use_magnet

            action = self._sample_action()
            self._random_toggle_mgt()

            self.env.step(action)
            s_t1 = self.env.get_observation()

            s_t1_q = np.array(s_t1["joints_angles"])

            delta_q = (s_t1_q - s_t_q).tolist()
            delta_mgt = self.env.robot.use_magnet - s_t_mgt

            step_collision = self._check_robot_distractor_collision()

            observations.append({
                "s_t": {
                    "joint_angles": s_t["joints_angles"],
                    "ee6D": s_t["endeff_6D"],
                    "obj6D": s_t["obj_6D"],
                    "occ": s_t["distractor"],
                    "mgt": s_t_mgt,
                },

                "a_t": {
                    "delta": delta_q,
                    "mgt": delta_mgt,
                },

                "s_t1": {
                    "joint_angles": s_t1["joints_angles"],
                    "ee6D": s_t1["endeff_6D"],
                    "obj6D": s_t1["obj_6D"],
                    "occ": s_t1["distractor"],
                    "mgt": self.env.robot.use_magnet,
                },

                "step_collision": step_collision
            })

        return observations