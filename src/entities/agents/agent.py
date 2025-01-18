from abc import ABC

from tensordict import TensorDict
from typing import List, Tuple, Union, Dict
from .base_agent import BaseAgent
from entities.features import Run
from os import makedirs, path
import torch
from torch.nn import ModuleDict
from torch.optim.adam import Adam


class Agent(BaseAgent):

    def __init__(self):
        self.networks: ModuleDict = ModuleDict()
        self.initialize_networks()
        self.optimizers: Dict[str, Adam] = dict()
        self.schedulers = dict()
        pass

    def initialize_networks(self):
        pass

    def act(
        self,
        state: torch.Tensor,
        return_dist: bool = False,
        test_phase: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.distributions.Normal]]]:
        means, stds = self.networks['actor'](state)
        batch_size = len(state)
        distributions = [torch.distributions.Normal(means[i], stds[i]) for i in range(batch_size)]
        if test_phase:
            action = [means[i] for i in range(batch_size)]
        else:
            action = [distributions[i].sample()[None, :] for i in range(batch_size)]
        if return_dist:
            return action, distributions
        return action

    def get_state_value(self, state) -> float:
        pass

    def save(self):
        experiment_path = Run.instance().experiment_path
        current_episode = Run.instance().dynamic_config.current_episode
        makedirs(f"{experiment_path}/networks/{current_episode}", exist_ok=True)
        torch.save(self.networks.state_dict(),
                   f"{experiment_path}/networks/{current_episode}/networks.pth")
        for name in self.optimizers.keys():
            torch.save(self.optimizers[name].state_dict(),
                       f"{experiment_path}/networks/{current_episode}/optimizer_{name}.pth")
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
        self.networks.load_state_dict(self.networks_state_dict)

        for name, optimizer in self.optimizers.items():
            optimizer_state_dict = torch.load(f"{load_path}/optimizer_{name}.pth")
            optimizer.load_state_dict(optimizer_state_dict)
