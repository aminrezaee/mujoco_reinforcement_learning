from .agent import Agent
from models.lstm.lstm_actor import LSTMActor as Actor
from models.lstm.lstm_critic import LSTMCritic as Critic
import torch
from entities.features import Run
from torch.optim.lr_scheduler import ExponentialLR
from typing import List, Tuple, Union


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

    def act(
        self,
        state: torch.Tensor,
        return_dist: bool = False,
        test_phase: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.distributions.Normal]]]:
        means, stds = self.networks['actor'](state)
        batch_size = len(state)
        distributions = torch.distributions.Normal(means, stds)
        if test_phase:
            action = [means[i] for i in range(batch_size)]
            action = torch.cat(action, dim=0)
        else:
            action = distributions.sample()
        if return_dist:
            return action, distributions
        return action
