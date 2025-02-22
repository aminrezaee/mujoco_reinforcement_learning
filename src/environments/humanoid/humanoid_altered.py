from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class SymmetricHumanoid(HumanoidEnv):

    def _get_rew(self, x_velocity, action):
        reward, reward_info = super()._get_rew(x_velocity, action)
        symmetric_reward = self.symmetric_reward() * 0.1
        reward_info["symmetric_reward"] = symmetric_reward
        reward = reward + symmetric_reward
        return reward, reward_info
        """
        The body parts are:

    | body part       | id (for `v2`, `v3`, `v4)` | id (for `v5`) |
    |  -------------  |  ---   |  ---  |
    | worldbody (note: all values are constant 0) | 0  |excluded|
    | torso           |1  | 0      |
    | lwaist          |2  | 1      |
    | pelvis          |3  | 2      |
    | right_thigh     |4  | 3      |
    | right_sin       |5  | 4      |
    | right_foot      |6  | 5      |
    | left_thigh      |7  | 6      |
    | left_sin        |8  | 7      |
    | left_foot       |9  | 8      |
    | right_upper_arm |10 | 9      |
    | right_lower_arm |11 | 10     |
    | left_upper_arm  |12 | 11     |
    | left_lower_arm  |13 | 12     |
        """

    def symmetric_reward(self):
        mass_offsets = self.data.cinert[1:, 6:8]  # 13 body parts, 2 mass offsets (x, y)
        symmetric_foot = -abs(mass_offsets[5, 1] + mass_offsets[8, 1]).sum()
        symmetric_upper_arm = -abs(mass_offsets[9, 1] + mass_offsets[11, 1]).sum()
        symmetric_lower_arm = -abs(mass_offsets[10, 1] + mass_offsets[12, 1]).sum()
        symmetric_thigh = -abs(mass_offsets[3, 1] + mass_offsets[6, 1]).sum()
        symmetric_pelvis = -abs(mass_offsets[2]).sum()
        return symmetric_foot + symmetric_upper_arm + symmetric_lower_arm + symmetric_thigh + symmetric_pelvis
