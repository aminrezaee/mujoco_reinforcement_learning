from .agent import Agent
from entities.features import Run
from models.lstm_actor import LSTMActor as Actor
from models.lstm_critic import LSTMCritic as Critic
import torch
from torch.optim.lr_scheduler import ExponentialLR


class LSTMAgent(Agent):

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
        means, stds = self.actor.act(state)
        # sub_action_count = Run.instance().agent_config.sub_action_count
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
