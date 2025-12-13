import numpy as np
from typing import Dict, Any, List

from mygym.myGym.envs.gym_env import GymEnv


def get_action_vector(
        s_t: Dict[str, Any],
        s_t1: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Returns an action vector given as the difference between joint angles and status of the robot's magnet of
    state s_t and s_t1.

    :param s_t: Previous state vector.
    :param s_t1: Current state vector.
    :return: Dictionary containing information about the action vector.
    """

    # Get joint angles and magnet status of previous state
    s_t_q = np.array(s_t["joint_angles"])
    s_t_mgt = s_t["mgt"]

    # Get joint angles and magnet status of current state
    s_t1_q = np.array(s_t1["joint_angles"])
    s_t1_mgt = s_t1["mgt"]

    # Action vector is given as difference between joint angles and magnet status
    delta_q = (s_t1_q - s_t_q).tolist()
    delta_mgt = s_t1_mgt - s_t_mgt

    return {
        "delta": delta_q,
        "mgt": delta_mgt,
    }


class GymWrapper(GymEnv):

    def __init__(
            self,
            cfg: Dict[str, Any],
            mgt_prob: float = 0.5
    ):
        """
        Class that adds additional functionality to the GymEnv class used during data generation.

        :param cfg: Configuration file for the environment.
        :param mgt_prob: Probability of random magnetization.
        """

        # Create base environment based on given config
        super().__init__(
            task_objects=cfg["task_objects"],
            observation=cfg["observation"],
            workspace=cfg["workspace"],
            dimension_velocity=cfg.get("dimension_velocity", 0.05),
            used_objects=cfg["used_objects"],
            action_repeat=cfg["action_repeat"],
            color_dict=cfg.get("color_dict", {}),
            robot=cfg["robot"],
            robot_action=cfg["robot_action"],
            max_velocity=cfg["max_velocity"],
            max_force=cfg["max_force"],
            robot_init_joint_poses=cfg["robot_init"],
            task_type=cfg["task_type"],
            num_networks=cfg.get("num_networks", 1),
            network_switcher=cfg.get("network_switcher", "gt"),
            distractors=cfg["distractors"],


            active_cameras=cfg["camera"],
            dataset=False,
            obs_space=None,
            visualize=cfg["visualize"],
            visgym=cfg["visgym"],
            logdir=cfg["logdir"],

            natural_language=bool(cfg["natural_language"]),
            training=True,
            top_grasp=False,

            gui_on=bool(cfg["gui"]),
            max_ep_steps=cfg["max_episode_steps"],
        )

        # Probability of magnet of the robot turning on
        self.mgt_prob = mgt_prob

        # For memorization of the initiated goal object for resets
        self._goal_init_pos = None
        self._goal_init_orn = None
        self._distractor_init_poses = None
        self.ee_init_pose = self.robot.get_position()

        # IDs of waypoint markers for visualization
        self._waypoint_marker_ids = []

    def _capture_initial_object_poses(
            self
    ) -> None:
        """
        Called once to remember where goal and distractors were
        right after env.reset(), so soft resets can put them back.
        """
        self._goal_init_pos = self.get_goal_position()
        self._goal_init_orn = self.get_goal_orientation()

        distractors = self.task_objects.get("distractor", [])
        self._distractor_init_poses = []
        for d in distractors:
            pos = np.array(d.get_position(), dtype=float)
            orn = np.array(d.get_orientation(), dtype=float)
            self._distractor_init_poses.append((pos, orn))

    def _is_goal_reachable(
            self
    ) -> bool:
        goal_pos = self.get_goal_position().astype(float)

        target_q = self.robot.calculate_accurate_IK(goal_pos)
        if target_q is None:
            return False

        self.robot.set_magnetization(1)
        self.step(target_q)

        return len(self.robot.magnetized_objects) > 0

    def reset_until_reachable(
            self,
            max_tries: int = 50
    ) -> Any:
        for i in range(max_tries):
            obs = self.reset()
            if self._is_goal_reachable():
                self.soft_reset_robot_only()
                return obs

        raise RuntimeError(f"Could not find a reachable goal after {max_tries} resets.")

    ##########################################
    ### Helper methods for data generation ###
    ##########################################

    def get_ef_position(
            self
    ) -> np.ndarray:
        """
        Used to determine the position of robot's end effector.

        :return: ([x,y,z]) Coordinates of end effector position.
        """

        return np.array(self.robot.get_position(), dtype=float)

    def get_goal_position(
            self
    ) -> np.ndarray:
        """
        Used to determine the position of the goal object.

        :return: ([x, y, z]) Coordinates of goal object.
        """
        goal_obj = self.task_objects["goal_state"]
        goal_obj_pos =  np.array(goal_obj.get_position(), dtype=float)

        return goal_obj_pos

    def get_goal_orientation(
            self
    ) -> np.ndarray:
        """
        Used to determine the orientation of the goal object.

        :return: (quaternion [x,y,z,w]) Orientation of goal object.
        """
        goal_obj = self.task_objects["goal_state"]
        goal_obj_orn = np.array(goal_obj.get_orientation(), dtype=float)

        return goal_obj_orn

    def get_distractors(
            self
    ) -> List[Any]:
        """
        Used to access list of distractors.

        :return: List of EnvObjects representing distractors.
        """
        return self.task_objects.get("distractor", [])

    def set_robot_configuration_kinematic(
            self,
            q: np.ndarray
    ) -> None:
        q = np.asarray(q, dtype=float)
        q = np.clip(q, self.robot.joints_limits[0], self.robot.joints_limits[1])
        for jid, idx in enumerate(self.robot.motor_indices):
            self.p.resetJointState(self.robot.robot_uid, idx, float(q[jid]))

    def random_toggle_mgt(
            self
    ) -> None:
        """
        Randomly turns on/off the magnet attached to the arm of the robot based on the given probability.

        Called to randomize the robot's behaviour during dataset generation.
        """
        if np.random.rand() < self.mgt_prob:
            self.robot.set_magnetization(1)
        else:
            self.robot.set_magnetization(0)

    def check_robot_distractor_collision(
            self
    ) -> bool:
        """
        Checks whether the robot collision with the distractors occurs provided by the presence of
        contact points of the robot and the distractors.

        Called during dataset generation to determine colliding states and actions that result in it,
        or collision trajectories.

        :return: True only if a collision has transpired (i.e. there is at least one contact point).
        """
        distractors = self.task_objects.get("distractor", None)
        if distractors is None:
            return False

        # Get contact points through the utilized physics engine
        for d in distractors:
            cps = self.p.getContactPoints(self.robot.get_uid(), d.get_uid())
            if len(cps) > 0:
                return True

        return False

    def soft_reset_robot_only(
            self
    ) -> Any:
        """
        Soft reset of the environment to generate multiple trajectories with same setup of the objects.

        If it is the first time the environment is going to be reset, we capture the placement of goal object and
        the distractors. Only robot and internal episode representation are reset to initial state. Goal object and
        distractors keep their original placement.

        As the distractors should be unmoveable, we freeze them in place after placing them following the reset.

        :return: Observation of the environment.
        """
        # Memorize initial poses the first time
        if self._goal_init_pos is None:
            self._capture_initial_object_poses()

        # Reset robot state
        self.robot.reset(random_robot=False)
        self.robot.set_magnetization(0)
        if hasattr(self.robot, "release_all_objects"):
            self.robot.release_all_objects()

        # Reset episode
        self.task.reset_task()

        # Put goal object back
        goal_obj = self.task_objects["goal_state"]
        self.p.resetBasePositionAndOrientation(
            goal_obj.get_uid(),
            self._goal_init_pos.tolist(),
            self._goal_init_orn.tolist()
        )

        # Put distractors back
        distractors = self.task_objects.get("distractor", [])
        if self._distractor_init_poses is not None:
            for d, (pos, orn) in zip(distractors, self._distractor_init_poses):
                self.p.resetBasePositionAndOrientation(
                    d.get_uid(),
                    pos.tolist(),
                    orn.tolist()
                )

        # Step once so the engine updates
        self.p.stepSimulation()
        obs = self.get_observation()

        # Freeze distractors in place by reducing their masses to 0
        distractors = self.task_objects.get("distractor", [])
        for d in distractors:
            uid = d.get_uid()
            # link index -1 = base
            self.p.changeDynamics(
                uid,
                -1,
                mass=0.0,  # static
                linearDamping=1.0,
                angularDamping=1.0
            )

        return obs

    ##################################################
    ### Debugging methods for visualization in GUI ###
    ##################################################

    def draw_executed_ee_trajectory_point(
            self,
            prev_pos: np.ndarray,
            curr_pos: np.ndarray,
            line_color: tuple[int, int, int]=(0, 1, 0),
            line_width: float=2.0,
            life_time: float=0.0,
    ) -> None:
        """
        Visualizes the executed movement of the end effector.

        :param prev_pos: Previous position of the end effector.
        :param curr_pos: Current position of the end effector.
        :param line_color: Colour of the line representing the movement.
        :param line_width: Width of the line representing the movement.
        :param life_time: Duration of the visualization. 0.0 -> lines stay until reset/removeAllUserDebugItems.
        """
        self.p.addUserDebugLine(
            prev_pos.tolist(),
            curr_pos.tolist(),
            line_color,
            lineWidth=line_width,
            lifeTime=life_time,
        )

    def show_waypoint_marker(
            self,
            pos: np.ndarray,
            size: float = 0.5,
            color: tuple[int, int, int]=(1, 1, 0),
            life_time: float=0.0,
    ) -> None:
        """
        Visualizes the waypoint marker of planned trajectory.

        :param pos: Position of the waypoint marker.
        :param size: Size of the marker.
        :param color: Colour of the marker.
        :param life_time: Duration of the visualization. 0.0 -> lines stay until reset/removeAllUserDebugItems.
        """
        x, y, z = pos.tolist()
        # Lines creating the cross representation of the marker
        lines = [
            # X-axis
            self.p.addUserDebugLine(
                [x - size, y, z],
                [x + size, y, z],
                lineColorRGB=color,
                lineWidth=2.0,
                lifeTime=life_time,
            ),
            # Y-axis
            self.p.addUserDebugLine(
                [x, y - size, z],
                [x, y + size, z],
                lineColorRGB=color,
                lineWidth=2.0,
                lifeTime=life_time,
            ),
            # Z-axis
            self.p.addUserDebugLine(
                [x, y, z - size],
                [x, y, z + size],
                lineColorRGB=color,
                lineWidth=2.0,
                lifeTime=life_time,
            )
        ]

        #Remember new marker for selective removal
        self._waypoint_marker_ids.extend(lines)

    def clear_waypoint_marker(
            self
    ) -> None:
        """
        Clears waypoint markers based on the waypoint marker ids.

        Clears memorized waypoint markers.
        """
        for uid in self._waypoint_marker_ids:
            self.p.removeUserDebugItem(uid)

        self._waypoint_marker_ids = []

    def draw_box(
            self,
            bounds: list[float],
            color: tuple[int, int, int]=(1, 0, 0),
            line_width: float=2.0,
            life_time: float=0.0,
    ) -> None:
        """
        Draw an axis-aligned bounding box using debug lines.

        :param bounds: ([x_min, x_max, y_min, y_max, z_min, z_max]) Geometry of the bounding box given through bounds.
        :param color: Color of the bounding box.
        :param line_width: Width of lines of the bounding box.
        :param life_time: Duration of the visualization. 0.0 -> lines stay until reset/removeAllUserDebugItems
        """
        # Bounds
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        # Corners of the box
        p000 = [x_min, y_min, z_min]
        p001 = [x_min, y_min, z_max]
        p010 = [x_min, y_max, z_min]
        p011 = [x_min, y_max, z_max]
        p100 = [x_max, y_min, z_min]
        p101 = [x_max, y_min, z_max]
        p110 = [x_max, y_max, z_min]
        p111 = [x_max, y_max, z_max]

        points = [p000, p001, p010, p011, p100, p101, p110, p111]

        # Edges of the box as (start_index, end_index) in points
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7),
            (6, 7),
        ]

        # Draw the box using its edges as debug lines
        for i, j in edges:
            self.p.addUserDebugLine(
                points[i],
                points[j],
                color,
                lineWidth=line_width,
                lifeTime=life_time,
            )

    ################################################################
    ### State and action representation methods for logging data ###
    ################################################################

    def get_state_vector(
            self
    ) -> Dict[str, Any]:
        """
        Based on the observation of the environment, returns a state vector containing:
            - joint angles of the robot
            - end effector description as position and rotation
            - goal object description as position and rotation
            - occlusions (distractors) present in the environment and their description given as position and rotation
            - state of the magnet (bool)

        :return: Dictionary containing information about the state vector.
        """
        state = self.get_observation()

        return {
            "joint_angles": state["additional_obs"]["joints_angles"],
            "ee6D": state["actual_state"],
            "obj6D": state["goal_state"],
            "occ": state["additional_obs"]["distractor"],
            "mgt": self.robot.use_magnet,
        }

