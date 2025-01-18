from .agent import Agent
from models.lstm_actor import LSTMActor as Actor
from models.lstm_critic import LSTMCritic as Critic
import torch
from entities.features import Run
from torch.optim.lr_scheduler import ExponentialLR
from os import makedirs, path
from torch.nn import ModuleDict


class PPOAgent(Agent):

    def initialize_networks(self):
        self.networks['actor'] = Actor()
        self.networks['critic'] = Critic()
        actor_optimizer = torch.optim.Adam(self.networks['actor'].parameters(),
                                           lr=Run.instance().training_config.learning_rate)
        critic_optimizer = torch.optim.Adam(self.networks['critic'].parameters(),
                                            lr=Run.instance().training_config.learning_rate)
        self.optimizers['actor'] = actor_optimizer
        self.optimizers['critic'] = critic_optimizer
        self.schedulers['actor'] = ExponentialLR(actor_optimizer, gamma=0.999)
        self.schedulers['critic'] = ExponentialLR(critic_optimizer, gamma=0.999)

    def get_state_value(self, state):
        return self.networks['critic'](state)
