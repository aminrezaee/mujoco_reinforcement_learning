from .agent import Agent
from models.lstm.lstm_actor import LSTMActor as Actor
from models.lstm.lstm_q_network import LSTMQNetwork as QNetwork
import torch
from entities.features import Run
from torch.optim.lr_scheduler import ExponentialLR
from os import path


class SoftActorCriticAgent(Agent):

    def initialize_networks(self):
        run = Run.instance()
        # initialize models
        self.networks['actor'] = Actor()
        self.networks['online_critic'] = QNetwork()
        self.networks['target_critic'] = QNetwork()
        # initialize optimizers
        self.optimizers['actor'] = torch.optim.Adam(self.networks['actor'].parameters(),
                                                    lr=run.training_config.learning_rate)
        self.optimizers['online_critic'] = torch.optim.Adam(
            self.networks['online_critic'].parameters(), lr=run.training_config.learning_rate)
        self.optimizers['target_critic'] = torch.optim.Adam(
            self.networks['target_critic'].parameters(), lr=run.training_config.learning_rate)
        if run.sac_config.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=run.device)
            self.optimizers['alpha'] = torch.optim.Adam(
                [self.log_alpha], lr=Run.instance().training_config.learning_rate)
        # initialize schedulers
        self.schedulers['actor'] = ExponentialLR(self.optimizers['actor'], gamma=0.999)
        self.schedulers['online_critic'] = ExponentialLR(self.optimizers['online_critic'],
                                                         gamma=0.999)
        self.schedulers['target_critic'] = ExponentialLR(self.optimizers['target_critic'],
                                                         gamma=0.999)
        return

    def get_state_value(self, state):
        return self.networks['critic'](state)
