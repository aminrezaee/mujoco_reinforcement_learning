from abc import ABC

from tensordict import TensorDict
from .base_agent import BaseAgent
from entities.features import Run
from os import makedirs, path
import torch
from torch.nn import Module


class Agent(BaseAgent):

    def __init__(self):
        pass

    def act(self, state, return_dist: bool = False):
        pass

    def train(self, memory: TensorDict):
        pass

    def get_state_value(self, state) -> float:
        pass

    def save(self):
        experiment_path = Run.instance().experiment_path
        current_episode = Run.instance().dynamic_config.current_episode
        makedirs(f"{experiment_path}/networks/{current_episode}", exist_ok=True)
        torch.save(self.actor.state_dict(),
                   f"{experiment_path}/networks/{current_episode}/actor.pth")
        torch.save(self.critic.state_dict(),
                   f"{experiment_path}/networks/{current_episode}/critic.pth")
        torch.save(self.actor_optimizer.state_dict(),
                   f"{experiment_path}/networks/{current_episode}/actor_optimizer.pth")
        torch.save(self.critic_optimizer.state_dict(),
                   f"{experiment_path}/networks/{current_episode}/critic_optimizer.pth")
        Run.instance().save()

    def load(self):
        experiment_path = Run.instance().experiment_path
        current_episode = Run.instance().dynamic_config.current_episode
        load_path = f"{experiment_path}/networks/{current_episode}"
        if not path.exists(load_path):
            load_path = f"{experiment_path}/networks/best_results/{current_episode}"
        if not path.exists(load_path):
            raise ValueError("the current iteration does not exist")
        self.actor_state_dict = torch.load(f"{load_path}/actor.pth")
        self.actor = Actor()
        self.actor.load_state_dict(self.actor_state_dict)

        self.critic_state_dict = torch.load(f"{load_path}/critic.pth")
        self.critic = Critic()
        self.critic.load_state_dict(self.critic_state_dict)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=Run.instance().training_config.learning_rate)
        self.actor_optimizer_state_dict = torch.load(f"{load_path}/actor_optimizer.pth")
        self.actor_optimizer.load_state_dict(self.actor_optimizer_state_dict)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=Run.instance().training_config.learning_rate)
        self.critic_optimizer_state_dict = torch.load(f"{load_path}/critic_optimizer.pth")
        self.critic_optimizer.load_state_dict(self.critic_optimizer_state_dict)
