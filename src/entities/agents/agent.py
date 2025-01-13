from abc import ABC

from tensordict import TensorDict
from .base_agent import BaseAgent
from entities.features import Run
from os import makedirs, path
import torch
from torch.nn import ModuleDict


class Agent(BaseAgent):

    def __init__(self):
        pass

    def act(self, state, return_dist: bool = False):
        pass

    def get_state_value(self, state) -> float:
        pass

    def save(self):
        experiment_path = Run.instance().experiment_path
        current_episode = Run.instance().dynamic_config.current_episode
        makedirs(f"{experiment_path}/networks/{current_episode}", exist_ok=True)
        torch.save(self.networks.state_dict(),
                   f"{experiment_path}/networks/{current_episode}/networks.pth")
        torch.save(self.optimizer.state_dict(),
                   f"{experiment_path}/networks/{current_episode}/optimizer.pth")
        Run.instance().save()

    def load(self):
        run = Run.instance()
        experiment_path = run.experiment_path
        current_episode = run.dynamic_config.current_episode
        load_path = f"{experiment_path}/networks/{current_episode}"
        if not path.exists(load_path):
            load_path = f"{experiment_path}/networks/best_results/{current_episode}"
        if not path.exists(load_path):
            raise ValueError("the current iteration does not exist")
        self.networks_state_dict = torch.load(f"{load_path}/networks.pth")
        self.networks = ModuleDict()
        self.networks.load_state_dict(self.networks_state_dict)

        self.optimizer = torch.optim.Adam(self.networks.parameters(),
                                          lr=run.training_config.learning_rate)
        self.optimizer_state_dict = torch.load(f"{load_path}/optimizer.pth")
        self.optimizer.load_state_dict(self.optimizer_state_dict)
