from entities.features import Run
from entities.agents.agent import Agent
from dataclasses import dataclass
import numpy as np
import torch
from tensordict import TensorDict
from utils.logger import Logger
import gymnasium as gym
from environments.helper import EnvironmentHelper as Helper


@dataclass
class Timestep:
    observation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict


class EnvironmentHelper(Helper):

    def initialize(self):
        self.environment = gym.vector.make("Humanoid-v4",
                                           render_mode="rgb_array",
                                           num_envs=self.run.environment_config.num_envs)
        self.test_environment = gym.make("Humanoid-v4", render_mode="rgb_array")
        self.timestep = Timestep(
            np.zeros((
                self.run.environment_config.num_envs,
                self.run.network_config.input_shape,
                self.run.environment_config.window_length,
            )), 0.0,
            np.zeros(self.run.environment_config.num_envs).astype(np.bool_),
            np.zeros(self.run.environment_config.num_envs).astype(np.bool_), {})
        self.test_timestep = Timestep(
            np.zeros((
                self.run.network_config.input_shape,
                self.run.environment_config.window_length,
            )), 0.0, False, False, {})

    def step(self, action: np.ndarray):
        """
        this function is called only for training phase

        Args:
            action (np.ndarray): action shape is (num_envs, action_shape)
        """
        last_timestep = Timestep(*self.environment.step(action))
        environment_timestep: Timestep = self.environment.timestep
        environment_timestep.reward = last_timestep.reward
        environment_timestep.terminated = last_timestep.terminated
        environment_timestep.truncated = last_timestep.truncated
        environment_timestep.info = last_timestep.info
        for i in range(Run.instance().environment_config.num_envs):
            if environment_timestep.terminated[i]:
                environment_timestep.observation[i, ...] = last_timestep.observation[i][:, None]
            else:
                self.shift_observations(test_phase=False, environment_index=i)
                environment_timestep.observation[i, :, -1] = last_timestep.observation[i, ...]
        self.rewards.append(self.environment.timestep.reward)

    def get_state(self, test_phase: bool) -> torch.Tensor:
        timestep: Timestep = self.get_using_environment(test_phase).timestep
        next_data = torch.tensor(timestep.observation)
        if self.run.normalize_observations:
            next_data = next_data - next_data.mean(dim=1).unsqueeze(1)
            std = next_data.std(dim=1).unsqueeze(1)
            std[std == 0] = 1e-8
            next_data = next_data / std
        next_data = next_data.to(Run.instance().dtype)
        if test_phase:
            next_data = next_data[None, :]
        next_data = next_data.permute(0, 2, 1)
        return next_data

    def reset_environment(self, test_phase):
        super().reset_environment(test_phase)
        if not test_phase:
            self.timestep.terminated = np.zeros(self.run.environment_config.num_envs).astype(
                np.bool_)
            self.timestep.truncated = np.zeros(self.run.environment_config.num_envs).astype(
                np.bool_)
