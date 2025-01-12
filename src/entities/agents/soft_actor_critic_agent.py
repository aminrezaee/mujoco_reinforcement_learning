from .agent import Agent
from entities.features import Run
from models.lstm_actor import LSTMActor as Actor
from models.lstm_critic import LSTMCritic as Critic
import torch
from tensordict import TensorDict
from utils.logger import Logger
from os import makedirs, path
import numpy as np
from torch.nn.functional import huber_loss
from torch.optim.lr_scheduler import ExponentialLR


class PPOAgent(Agent):

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.run = Run.instance()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.run.training_config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.run.training_config.learning_rate)
        self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=0.999)
        self.critic_scheduler = ExponentialLR(self.critic_optimizer, gamma=0.999)

    def act(self,
            state: torch.Tensor,
            return_dist: bool = False,
            test_phase: bool = False) -> torch.Tensor:
        pass

    def get_state_value(self, state):
        pass

    def train(self, memory: TensorDict):
        pass
