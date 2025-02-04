from abc import ABC
import numpy as np
import torch
from entities.features import Run
from os import makedirs
import mediapy as media
from entities.timestep import Timestep
from gymnasium.core import Env
import mlflow


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

    def reset(self, release_memory: bool = True):
        self.rewards = []
        if release_memory:
            self.memory = []
        self.images = []

    def visualize(self):
        run = Run.instance()
        path = f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}"
        makedirs(path, exist_ok=True)
        media.write_video(f"{path}/video.gif", self.images, fps=30, codec="gif")
        mlflow.log_artifact(f"{path}/video.gif")

    def get_using_environment(self, test_phase: bool) -> Env:
        using_environment = self.environment
        if test_phase:
            using_environment = self.test_environment
        return using_environment

    def shift_observations(self, test_phase: bool, environment_index: int):
        timestep: Timestep = self.get_using_environment(test_phase).timestep
        if test_phase:
            timestep.observation[..., :-1] = timestep.observation[..., 1:]
        else:
            timestep.observation[environment_index, :, :-1] = timestep.observation[
                environment_index, :, 1:]

    def reset_environment(self, test_phase: bool):
        using_environment = self.get_using_environment(test_phase)

        last_observation, using_environment.timestep.info = using_environment.reset()
        using_environment.timestep.observation = np.repeat(
            last_observation[..., None], axis=-1, repeats=self.run.environment_config.window_length)
        if test_phase:
            self.test_timestep.terminated = False
            self.test_timestep.truncated = False
