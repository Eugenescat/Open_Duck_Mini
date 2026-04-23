import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

from mini_bdx.utils.mujoco_utils import check_contact, get_contact_force

FRAME_SKIP = 4


class BDXEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 125,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        self.nb_dofs = 15

        observation_space = Box(
            np.array(
                [
                    *(-np.pi * np.ones(self.nb_dofs)),
                    *(-10 * np.ones(self.nb_dofs)),
                    *(-10 * np.ones(3)),
                    *(-10 * np.ones(3)),
                    *(-10 * np.ones(3)),
                    *(0 * np.ones(2)),
                    *(-np.pi * np.ones(2)),
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    *(np.pi * np.ones(self.nb_dofs)),
                    *(10 * np.ones(self.nb_dofs)),
                    *(10 * np.ones(3)),
                    *(10 * np.ones(3)),
                    *(10 * np.ones(3)),
                    *(1 * np.ones(2)),
                    *(np.pi * np.ones(2)),
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.right_foot_contact = True
        self.left_foot_contact = True
        self.prev_action = np.zeros(self.nb_dofs)
        self.prev_torque = np.zeros(self.nb_dofs)
        self.prev_t = 0
        self.startup_cooldown = 1.0
        self.walk_period = 1.0
        self.target_velocities = np.asarray([0.20, 0.0, 0.0])
        self.cumulated_reward = 0.0
        self.prev_base_x = 0.0
        self.prev_left_contact = False
        self.prev_right_contact = False
        self.prev_left_foot_x = 0.0
        self.prev_right_foot_x = 0.0
        self.prev_support_side = 0
        self.episode_start_x = 0.0

        self.init_pos = np.array(
            [
                -0.013946457213457239,
                0.07918837709879874,
                0.5325073962634973,
                -1.6225192902713386,
                0.9149246381274986,
                0.013627156377842975,
                0.07738878096596595,
                0.5933527914082196,
                -1.630548419252953,
                0.8621333440557593,
                -0.17453292519943295,
                -0.17453292519943295,
                8.65556854322817e-27,
                0,
                0,
            ]
        )

        MujocoEnv.__init__(
            self,
            "../../../mini_bdx/robots/bdx/scene.xml",
            FRAME_SKIP,
            observation_space=observation_space,
            **kwargs,
        )

    def is_terminated(self) -> bool:
        left_antenna_contact = check_contact(
            self.data, self.model, "left_antenna_assembly", "floor"
        )
        right_antenna_contact = check_contact(
            self.data, self.model, "right_antenna_assembly", "floor"
        )
        body_contact = check_contact(self.data, self.model, "body_module", "floor")
        rot = np.array(self.data.body("base").xmat).reshape(3, 3)
        z_vec = rot[:, 2]
        z_vec /= np.linalg.norm(z_vec)
        upright = np.array([0, 0, 1])
        roll, pitch = self.base_rpy()
        return (
            self.data.body("base").xpos[2] < 0.08
            or np.dot(upright, z_vec) <= 0.4
            or abs(roll) > 0.55
            or abs(pitch) > 0.50
            or left_antenna_contact
            or right_antenna_contact
            or body_contact
        )

    def get_clock_signal(self):
        a = np.sin(2 * np.pi * (self.data.time % self.walk_period) / self.walk_period)
        b = np.cos(2 * np.pi * (self.data.time % self.walk_period) / self.walk_period)
        return [a, b]

    def follow_xy_target_reward(self):
        x_velocity = self.data.body("base").cvel[3:][0]
        y_velocity = self.data.body("base").cvel[3:][1]
        x_error = abs(self.target_velocities[0] - x_velocity)
        y_error = abs(self.target_velocities[1] - y_velocity)
        return -(x_error + y_error)

    def forward_velocity_reward(self):
        x_velocity = self.data.body("base").cvel[3:][0]
        return np.exp(-30.0 * (x_velocity - self.target_velocities[0]) ** 2)

    def forward_progress_reward(self):
        x_now = float(self.data.body("base").xpos[0])
        dx = x_now - self.prev_base_x
        # Encourage actual displacement in the world frame.
        return np.clip(dx * 40.0, -1.0, 1.0)

    def forward_displacement_reward(self):
        x_now = float(self.data.body("base").xpos[0])
        disp = x_now - self.episode_start_x
        return np.clip(disp * 0.35, 0.0, 1.0)

    def low_forward_speed_penalty(self):
        x_velocity = float(self.data.body("base").cvel[3:][0])
        # Penalize near-zero or backward behavior when trying to walk.
        return -np.clip((0.08 - x_velocity) * 4.0, 0.0, 1.0)

    def base_rpy(self):
        quat = np.array(self.data.body("base").xquat)
        # mujoco xquat is [w, x, y, z], scipy expects [x, y, z, w]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        e = r.as_euler("xyz", degrees=False)
        return float(e[0]), float(e[1])

    def pitch_penalty(self):
        # Penalize forward/backward leaning, especially larger angles.
        _, pitch = self.base_rpy()
        return -np.square(pitch / 0.30)

    def roll_penalty(self):
        roll, _ = self.base_rpy()
        return -np.square(roll / 0.28)

    def forward_gate(self):
        # Relaxed gate: only require upright/height/roll/pitch, NOT single-support.
        # The original strict gate set g=0 whenever the robot stood on both feet,
        # which created a stand-still local optimum (the agent learned to just
        # stand forever, earning ~0.34/step from posture terms). Letting forward
        # rewards flow to standing upright robots forces the agent to actually
        # move to maximize return.
        h = float(self.data.body("base").xpos[2])
        roll, pitch = self.base_rpy()
        upright = self.upright_reward()
        if h < 0.13 or abs(roll) > 0.25 or abs(pitch) > 0.24 or upright < 0.55:
            return 0.0
        return 1.0

    def support_side(self):
        left = bool(self.left_foot_contact)
        right = bool(self.right_foot_contact)
        if left and (not right):
            return -1  # left support
        if right and (not left):
            return 1  # right support
        return 0  # double support or flight

    def lateral_stability_reward(self):
        y_velocity = self.data.body("base").cvel[3:][1]
        return -abs(y_velocity)

    def yaw_stability_reward(self):
        yaw_velocity = self.data.body("base").cvel[:3][2]
        return -abs(yaw_velocity - self.target_velocities[2])

    def follow_yaw_target_reward(self):
        yaw_velocity = self.data.body("base").cvel[:3][2]
        yaw_error = abs(self.target_velocities[2] - yaw_velocity)
        return -yaw_error

    def height_reward(self):
        current_height = self.data.body("base").xpos[2]
        return np.exp(-40 * (0.15 - current_height) ** 2)

    def upright_reward(self):
        z_vec = np.array(self.data.body("base").xmat).reshape(3, 3)[:, 2]
        return np.square(np.dot(np.array([0, 0, 1]), z_vec))

    def torque_reward(self):
        current_torque = self.data.qfrc_actuator
        return np.exp(
            -0.25 * np.sum((self.prev_torque - current_torque) ** 2) / self.nb_dofs
        )

    def _reference_pose(self):
        phase = 2 * np.pi * (self.data.time % self.walk_period) / self.walk_period
        right_phase = np.sin(phase)
        left_phase = np.sin(phase + np.pi)
        ref = self.init_pos.copy()
        # Right leg swing/stance pattern.
        ref[2] += 0.20 * right_phase
        ref[3] += -0.35 * max(0.0, right_phase)
        ref[4] += 0.15 * max(0.0, right_phase)
        # Left leg mirrored.
        ref[7] += 0.20 * left_phase
        ref[8] += -0.35 * max(0.0, left_phase)
        ref[9] += 0.15 * max(0.0, left_phase)
        # Keep head fixed.
        ref[10:] = self.init_pos[10:]
        return ref

    def imitation_pose_reward(self):
        ref = self._reference_pose()
        current = self.data.qpos[7 : 7 + self.nb_dofs]
        err = np.mean(np.square(current[:10] - ref[:10]))
        return np.exp(-25.0 * err)

    def imitation_action_reward(self, a):
        ref = self._reference_pose()
        err = np.mean(np.square(a[:10] - ref[:10]))
        return np.exp(-20.0 * err)

    def support_flying_reward(self):
        right_contact_force = abs(
            np.sum(get_contact_force(self.data, self.model, "right_foot", "floor"))
        )
        left_contact_force = abs(
            np.sum(get_contact_force(self.data, self.model, "left_foot", "floor"))
        )
        right_speed = np.linalg.norm(self.data.body("right_foot").cvel[3:])
        left_speed = np.linalg.norm(self.data.body("left_foot").cvel[3:])
        imbalance = abs(left_contact_force - right_contact_force) + abs(
            right_speed - left_speed
        )
        # Lower imbalance should be better.
        return np.exp(-1.0 * imbalance)

    def contact_switch_reward(self):
        side = self.support_side()
        switched_support = (
            side != 0 and self.prev_support_side != 0 and side != self.prev_support_side
        )
        if switched_support:
            return 1.0
        return 0.0

    def single_support_reward(self):
        return 1.0 if self.support_side() != 0 else 0.0

    def double_support_penalty(self):
        return -1.0 if (self.left_foot_contact and self.right_foot_contact) else 0.0

    def swing_foot_forward_reward(self):
        side = self.support_side()
        right_x = float(self.data.body("right_foot").xpos[0])
        left_x = float(self.data.body("left_foot").xpos[0])

        # Reward forward movement of the swing foot under single support.
        if side == -1:  # left support -> right swings
            dx = right_x - self.prev_right_foot_x
            return np.clip(dx * 120.0, 0.0, 1.0)
        if side == 1:  # right support -> left swings
            dx = left_x - self.prev_left_foot_x
            return np.clip(dx * 120.0, 0.0, 1.0)
        return 0.0

    def step(self, a):
        t = self.data.time
        dt = t - self.prev_t
        if self.startup_cooldown > 0:
            self.startup_cooldown -= dt
            self.do_simulation(self.init_pos, FRAME_SKIP)
            reward = 0
        else:
            self.right_foot_contact = check_contact(
                self.data, self.model, "right_foot", "floor"
            )
            self.left_foot_contact = check_contact(
                self.data, self.model, "left_foot", "floor"
            )
            a += self.init_pos
            current_ctrl = self.data.ctrl.copy()
            delta_max = 0.05
            a = np.clip(a, current_ctrl - delta_max, current_ctrl + delta_max)
            a[10:] = self.init_pos[10:]
            self.do_simulation(a, FRAME_SKIP)
            gate = self.forward_gate()
            reward = (
                0.45 * gate * self.forward_velocity_reward()
                + 0.35 * gate * self.forward_progress_reward()
                + 0.15 * self.forward_displacement_reward()
                + 0.12 * self.swing_foot_forward_reward()
                + 0.05 * self.contact_switch_reward()
                + 0.04 * self.single_support_reward()
                + 0.10 * gate * self.low_forward_speed_penalty()
                + 0.06 * self.height_reward()
                + 0.06 * self.upright_reward()
                + 0.02 * self.lateral_stability_reward()
                + 0.02 * self.yaw_stability_reward()
                + 0.02 * self.torque_reward()
                + 0.02 * self.support_flying_reward()
                + 0.02 * self.double_support_penalty()
                + 0.03 * self.imitation_pose_reward()
                + 0.01 * self.imitation_action_reward(a)
                + 0.06 * self.pitch_penalty()
                + 0.04 * self.roll_penalty()
            )
            self.cumulated_reward += reward

        ob = self._get_obs()
        if self.render_mode == "human":
            self.render()
        self.prev_t = t
        self.prev_base_x = float(self.data.body("base").xpos[0])
        self.prev_action = a.copy()
        self.prev_torque = self.data.qfrc_actuator.copy()
        self.prev_left_contact = bool(self.left_foot_contact)
        self.prev_right_contact = bool(self.right_foot_contact)
        self.prev_left_foot_x = float(self.data.body("left_foot").xpos[0])
        self.prev_right_foot_x = float(self.data.body("right_foot").xpos[0])
        side = self.support_side()
        if side != 0:
            self.prev_support_side = side
        return (ob, reward, self.is_terminated(), False, {})

    def reset_model(self):
        self.prev_t = self.data.time
        self.startup_cooldown = 1.0
        self.cumulated_reward = 0.0
        self.target_velocities = np.asarray([0.20, 0.0, 0.0])
        self.prev_action = np.zeros(self.nb_dofs)
        self.prev_torque = np.zeros(self.nb_dofs)
        self.prev_left_contact = False
        self.prev_right_contact = False
        self.prev_support_side = 0
        self.goto_init()
        self.prev_base_x = float(self.data.body("base").xpos[0])
        self.episode_start_x = self.prev_base_x
        self.prev_left_foot_x = float(self.data.body("left_foot").xpos[0])
        self.prev_right_foot_x = float(self.data.body("right_foot").xpos[0])
        self.set_state(self.data.qpos, self.data.qvel)
        return self._get_obs()

    def goto_init(self):
        self.data.qvel[:] = np.zeros(len(self.data.qvel[:]))
        self.data.qpos[7 : 7 + self.nb_dofs] = self.init_pos
        self.data.qpos[2] = 0.15
        self.data.qpos[3 : 3 + 4] = [1, 0, 0.08, 0]
        self.data.ctrl[:] = self.init_pos

    def _get_obs(self):
        joints_rotations = self.data.qpos[7 : 7 + self.nb_dofs]
        joints_velocities = self.data.qvel[6 : 6 + self.nb_dofs]
        angular_velocity = self.data.body("base").cvel[:3]
        linear_velocity = self.data.body("base").cvel[3:]
        return np.concatenate(
            [
                joints_rotations,
                joints_velocities,
                angular_velocity,
                linear_velocity,
                self.target_velocities,
                [self.left_foot_contact, self.right_foot_contact],
                self.get_clock_signal(),
            ],
            dtype=np.float32,
        )
