from entities.features import Run
from entities.agents.agent import Agent
from dataclasses import dataclass
import numpy as np
import torch
from tensordict import TensorDict
from utils.logger import Logger
from torchrl.objectives.value.functional import generalized_advantage_estimate
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
        self.timestep = Timestep(np.zeros(self.run.network_config.input_shape), 0.0,
                                 np.zeros(self.run.environment_config.num_envs).astype(np.bool_),
                                 np.zeros(self.run.environment_config.num_envs).astype(np.bool_),
                                 {})
        self.test_timestep = Timestep(np.zeros(self.run.network_config.input_shape), 0.0, False,
                                      False, {})

    def reset_environment(self, test_phase):
        super().reset_environment(test_phase)
        if not test_phase:
            self.timestep.terminated = np.zeros(self.run.environment_config.num_envs).astype(
                np.bool_)
            self.timestep.truncated = np.zeros(self.run.environment_config.num_envs).astype(
                np.bool_)
