from .agent import Agent
from entities.features import Run
from models.lstm_actor import LSTMActor as Actor
from models.lstm_critic import LSTMCritic as Critic
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import ModuleDict


class LSTMAgent(Agent):

    def __init__(self):
        self.networks: ModuleDict = ModuleDict()
        self.optimizer = torch.optim.Adam(self.networks.parameters(),
                                          lr=Run.instance().training_config.learning_rate)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.999)

    def act(self,
            state: torch.Tensor,
            return_dist: bool = False,
            test_phase: bool = False) -> torch.Tensor:
        means, stds = self.actor.act(state)
        batch_size = len(state)
        distributions = [torch.distributions.Normal(means[i], stds[i]) for i in range(batch_size)]
        if test_phase:
            action = [means[i] for i in range(batch_size)]
        else:
            action = [distributions[i].sample()[None, :] for i in range(batch_size)]
        if return_dist:
            return action, distributions
        return action

    def get_state_value(self, state):
        return self.critic(state)
