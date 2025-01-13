from .agent import Agent
from entities.features import Run
from models.lstm_actor import LSTMActor as Actor
from models.lstm_critic import LSTMCritic as Critic
import torch
from tensordict import TensorDict
from utils.logger import Logger
from os import makedirs, path
import numpy as np
from torch.nn.functional import huber_loss, mse_loss
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

    def train(self, memory: TensorDict):
        batch_size = self.run.training_config.batch_size
        epochs = self.run.training_config.epochs_per_iteration
        batches_per_epoch = int(self.run.environment_config.maximum_timesteps *
                                self.run.environment_config.num_envs / batch_size)
        memory = memory.view(-1)
        epoch_losses = [[], []]
        for _ in range(epochs):
            iteration_losses = [[], []]
            idx = torch.randperm(len(memory))
            shuffled_memory = memory[idx]
            for i in range(batches_per_epoch):
                batch = shuffled_memory[int(i * batch_size):int((i + 1) * batch_size)]
                if len(batch) != batch_size:
                    continue
                sub_actions = batch['action']
                joint_index = np.random.randint(0, Run.instance().agent_config.sub_action_count)
                mean, std = self.actor(batch['current_state'])
                distributions = [
                    torch.distributions.Normal(mean[i], std[i]) for i in range(batch_size)
                ]
                action_log_prob = batch['action_log_prob'][:, joint_index]

                new_action_log_prob = torch.cat([
                    distributions[i].log_prob(sub_actions[i]).sum()[None] for i in range(batch_size)
                ])
                # critic loss
                current_state_value = self.get_state_value(batch['current_state'])
                current_state_value_target = batch['current_state_value_target']
                critic_loss: torch.Tensor = mse_loss(current_state_value,
                                                     current_state_value_target,
                                                     reduction='mean')
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),
                                               Run.instance().ppo_config.max_grad_norm)
                self.critic_optimizer.step()

                # actor loss
                advantage = batch['advantage']
                total_entropy = sum([d.entropy().mean() for d in distributions])
                ratio = (new_action_log_prob - action_log_prob).exp()[:, None]
                # print(ratio.min() , ratio.max())
                surrogate1 = ratio * advantage
                surrogate2 = torch.clamp(ratio, 1.0 - Run.instance().ppo_config.clip_epsilon,
                                         1.0 + Run.instance().ppo_config.clip_epsilon) * advantage
                actor_loss: torch.Tensor = -torch.min(surrogate1, surrogate2).mean(
                ) - total_entropy * Run.instance().ppo_config.entropy_eps
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                                               Run.instance().ppo_config.max_grad_norm)
                self.actor_optimizer.step()

                iteration_losses[0].append(actor_loss.detach().item())
                iteration_losses[1].append(critic_loss.detach().item())

            epoch_losses[0].append(sum(iteration_losses[0]) / len(iteration_losses[0]))
            epoch_losses[1].append(sum(iteration_losses[1]) / len(iteration_losses[1]))
        episode_actor_loss = sum(epoch_losses[0]) / len(epoch_losses[0])
        episode_critic_loss = sum(epoch_losses[1]) / len(epoch_losses[1])
        if self.run.dynamic_config.current_episode < 2500:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        Logger.log(
            f"Actor Loss: {episode_actor_loss} Critic Loss: {episode_critic_loss} Epoch Loss: {episode_actor_loss + episode_critic_loss}",
            episode=Run.instance().dynamic_config.current_episode,
            log_type=Logger.TRAINING_TYPE,
            print_message=True)
        pass
