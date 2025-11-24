import numpy as np
from typing import List, Dict, Any, Tuple

from myGym.myGym.envs.gym_env import GymEnv


class TrajectoryDatasetGenerator:

    def __init__(
            self,
            env: GymEnv,
            waypoint_sampling_space: List[int] = [-1, 1, -1, 1, 0, 1.3],
            points_per_segment: int = 10,
            n_trajectories: int = 20,
    ):
        self.env = env
        self.points_per_segment = points_per_segment
        self.n_trajectories = n_trajectories

        # sample space of waypoints given as array [x_min, x_max, y_min, y_max, z_min, z_max]
        self.waypoint_sampling_space = waypoint_sampling_space
        self.x_max = self.waypoint_sampling_space[1]
        self.y_max = self.waypoint_sampling_space[3]
        self.z_max = self.waypoint_sampling_space[5]
        self.x_min = self.waypoint_sampling_space[0]
        self.y_min = self.waypoint_sampling_space[2]
        self.z_min = self.waypoint_sampling_space[4]

    def _get_ef_position(
            self
    ) -> np.ndarray:
        return np.array(self.env.robot.get_position())

    def _get_goal_position(
            self
    ) -> np.ndarray:
        goal_obj = self.env.task_objects["goal_state"]
        return np.array(goal_obj.get_position())

    def _get_distractor_position(
            self
    ) -> np.ndarray:
        distractors = self.env.task_objects.get("distractor")
        d = distractors[0]

        return np.array(d.get_position())

    def _sample_waypoint(
        self,
    ) -> np.ndarray:

        x = (self.x_max - self.x_min) * np.random.random() + self.x_min
        y = (self.y_max - self.y_min) * np.random.random() + self.y_min
        z = (self.z_max - self.z_min) * np.random.random() + self.z_min

        return np.array([x, y, z], dtype=float)

    def _plan_trajectory_skeleton(
        self
    ) -> List[np.ndarray]:
        ef_start = self._get_ef_position()
        ef_goal = self._get_goal_position()

        waypoint = self._sample_waypoint()

        return [ef_start, waypoint, ef_goal]

    def _interpolate_segment(
        self,
        p0: np.ndarray,
        p1: np.ndarray
    ) -> List[np.ndarray]:
        if self.points_per_segment <= 1:
            return [p1.copy()]
        alphas = np.linspace(0.0, 1.0, self.points_per_segment, endpoint=True)
        return [p0 + a * (p1 - p0) for a in alphas]

    def _waypoints_to_joint_trajectory(
        self,
        ef_waypoints: List[np.ndarray]
    ) -> List[np.ndarray]:
        joint_traj: List[np.ndarray] = []

        for i in range(len(ef_waypoints) - 1):
            origin = ef_waypoints[i]
            goal = ef_waypoints[i + 1]
            segment_points = self._interpolate_segment(origin, goal)

            for index, point in enumerate(segment_points):
                q = np.array(self.env.robot._calculate_joint_poses(point.tolist()), dtype=float)

                if (i > 0) and (index == 0):
                    continue

                joint_traj.append(q)

        return joint_traj

    def _check_robot_distractor_collision(
            self
    ) -> bool:
        distractors = self.env.task_objects.get("distractor", None)
        if distractors is None:
            return False

        for d in distractors:
            cps = self.env.p.getContactPoints(self.env.robot.get_uid(), d.get_uid())
            if len(cps) > 0:
                return True

        return False

    @staticmethod
    def _soft_reset_robot_only(
            env: GymEnv
    ) -> Any:
        env.robot.reset(random_robot=False)
        env.episode_steps = 0
        env.episode_reward = 0.0
        env.task.reset_task()
        env.p.stepSimulation()
        obs = env.get_observation()
        return obs

    def run_planned_episode(
        self
    ) -> Dict[str, Any]:
        self._soft_reset_robot_only(env=self.env)
        self.env.robot.set_magnetization(0)

        ef_waypoints = self._plan_trajectory_skeleton()
        joint_traj = self._waypoints_to_joint_trajectory(ef_waypoints)

        episode_transitions = []
        episode_collision = False

        for q_target in joint_traj:
            s_t = self.env.get_observation()
            theta_t = np.array(s_t["joints_angles"])

            action = q_target

            self.env.step(action)
            s_t1 = self.env.get_observation()

            theta_t1 = np.array(s_t1["joints_angles"])

            delta_theta = (theta_t1 - theta_t).tolist()

            a_t = {
                "delta": delta_theta,
            }

            episode_transitions.append(
                {
                    "s_t": {
                        "joint_angles": s_t["joints_angles"],
                        "ee6D": s_t["endeff_6D"],
                        "obj6D": s_t["obj_6D"],
                        "occ": s_t["distractor"],
                        "mgt": self.env.robot.use_magnet,
                    },
                    "a_t": a_t,
                    "s_t1": {
                        "joint_angles": s_t1["joints_angles"],
                        "ee6D": s_t1["endeff_6D"],
                        "obj6D": s_t1["obj_6D"],
                        "occ": s_t1["distractor"],
                        "mgt": self.env.robot.use_magnet,
                    }
                }
            )

            if self._check_robot_distractor_collision():
                episode_collision = True
                break

        return {
            "transitions": episode_transitions,
            "episode_collision": episode_collision,
        }

    def collect_data(
        self
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        collision_episodes = []
        avoidance_episodes = []

        for _ in range(self.n_trajectories):
            ep = self.run_planned_episode()

            if ep["episode_collision"]:
                collision_episodes.append(ep)
            else:
                avoidance_episodes.append(ep)

        return collision_episodes, avoidance_episodes