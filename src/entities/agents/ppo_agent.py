from .agent import Agent
from models.lstm_actor import LSTMActor as Actor
from models.lstm_critic import LSTMCritic as Critic
import torch


class PPOAgent(Agent):

    def initialize_networks(self):
        self.networks['actor'] = Actor()
        self.networks['critic'] = Critic()

    def get_state_value(self, state):
        return self.networks['critic'](state)
