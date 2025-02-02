from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class Humanoid(HumanoidEnv):

    def __init__(self,
                 xml_file="humanoid.xml",
                 frame_skip=5,
                 default_camera_config=...,
                 forward_reward_weight=1.25,
                 ctrl_cost_weight=0.1,
                 contact_cost_weight=5e-7,
                 contact_cost_range=...,
                 healthy_reward=5,
                 terminate_when_unhealthy=True,
                 healthy_z_range=...,
                 reset_noise_scale=0.01,
                 exclude_current_positions_from_observation=True,
                 include_cinert_in_observation=True,
                 include_cvel_in_observation=True,
                 include_qfrc_actuator_in_observation=True,
                 include_cfrc_ext_in_observation=True,
                 **kwargs):
        super().__init__(xml_file, frame_skip, default_camera_config, forward_reward_weight,
                         ctrl_cost_weight, contact_cost_weight, contact_cost_range, healthy_reward,
                         terminate_when_unhealthy, healthy_z_range, reset_noise_scale,
                         exclude_current_positions_from_observation, include_cinert_in_observation,
                         include_cvel_in_observation, include_qfrc_actuator_in_observation,
                         include_cfrc_ext_in_observation, **kwargs)

    def _get_rew(self, x_velocity, action):
        original_reward = super()._get_rew(x_velocity, action)
        assymetric_reward = self.assymetric_reward()
        self.data
        return original_reward + assymetric_reward

    def assymetric_reward(self):
        return 0
