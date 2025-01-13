from abc import ABC
import numpy as np
import torch
from entities.features import Run
from os import makedirs
import mediapy as media
from entities.timestep import Timestep


class EnvironmentHelper(ABC):

    def __init__(self):
        self.rewards = []
        self.memory = []
        self.images = []
        self.run = Run.instance()
        self.initialize()
        self.environment.timestep = self.timestep
        self.test_environment.timestep = self.test_timestep

    def initialize(self):
        pass

    def step(self, action: np.ndarray):
        pass

    def get_state(self, test_phase: bool) -> torch.Tensor:
        pass

    def reset(self):
        self.rewards = []
        self.memory = []
        self.images = []

    def visualize(self):
        run = Run.instance()
        path = f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}"
        makedirs(path, exist_ok=True)
        media.write_video(f"{path}/video.mp4", self.images, fps=30)

    def get_using_environment(self, test_phase: bool):
        using_environment = self.environment
        if test_phase:
            using_environment = self.test_environment
        return using_environment

    def reset_environment(self, test_phase: bool):
        using_environment = self.get_using_environment(test_phase)

        last_observation, using_environment.timestep.info = using_environment.reset()
        using_environment.timestep.observation = np.repeat(
            last_observation[..., None], axis=-1, repeats=self.run.environment_config.window_length)
        if test_phase:
            self.test_timestep.terminated = False
            self.test_timestep.truncated = False
