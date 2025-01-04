from .agent import Agent
from entities.features import Run
from models.actor import Actor
from models.critic import Critic
import torch
from tensordict import TensorDict
from utils.logger import Logger
from os import makedirs
import numpy as np
from torch.nn.functional import huber_loss


class PPOAgent(Agent):

    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=Run.instance().training_config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=Run.instance().training_config.learning_rate)

    def act(self,
            state: torch.Tensor,
            return_dist: bool = False,
            test_phase: bool = False) -> torch.Tensor:
        means, stds = self.actor.act(state)
        sub_action_count = Run.instance().agent_config.sub_action_count
        distributions = [
            torch.distributions.Normal(means[i], stds[i]) for i in range(sub_action_count)
        ]
        if test_phase:
            action = [means[i] for i in range(sub_action_count)]
        else:
            action = [distributions[i].sample() for i in range(sub_action_count)]
        if return_dist:
            return action, distributions
        return action

    def get_state_value(self, state):
        return self.critic(state)

    def train(self, memory: TensorDict):
        batch_size = Run.instance().training_config.batch_size
        batches_per_epoch = Run.instance().training_config.batches_per_epoch
        epochs = Run.instance().training_config.epochs_per_iteration
        epoch_losses = [[], []]
        for _ in range(epochs):
            iteration_losses = [[], []]
            for _ in range(batches_per_epoch):
                idx = torch.randperm(len(memory))
                shuffled_memory = memory[idx]
                batch = shuffled_memory[:batch_size]
                sub_actions = batch['action']
                joint_index = np.random.randint(0, Run.instance().agent_config.sub_action_count)
                mean, std = self.actor(batch['current_state'], joint_index)
                distributions = [
                    torch.distributions.Normal(mean[i], std[i]) for i in range(batch_size)
                ]
                action_log_prob = batch['action_log_prob'][:, joint_index]

                new_action_log_prob = torch.cat([
                    distributions[i].log_prob(sub_actions[i, joint_index]).sum()[None]
                    for i in range(batch_size)
                ])
                # critic loss
                current_state_value = self.get_state_value(batch['current_state'])
                current_state_value_target = batch['current_state_value_target']
                critic_loss: torch.Tensor = huber_loss(current_state_value,
                                                       current_state_value_target,
                                                       reduction='mean')
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), Run.instance().ppo_config.max_grad_norm)
                self.critic_optimizer.step()

                # actor loss
                advantage = batch['advantage']
                total_entropy = sum([d.entropy().mean() for d in distributions])
                ratio = (new_action_log_prob - action_log_prob).exp()[:, None]
                surrogate1 = ratio * advantage
                surrogate2 = torch.clamp(ratio, 1.0 - Run.instance().ppo_config.clip_epsilon,
                                         1.0 + Run.instance().ppo_config.clip_epsilon) * advantage
                actor_loss: torch.Tensor = -torch.min(surrogate1, surrogate2).mean() - total_entropy * Run.instance().ppo_config.entropy_eps
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), Run.instance().ppo_config.max_grad_norm)
                self.actor_optimizer.step()

                iteration_losses[0].append(actor_loss.item())
                iteration_losses[1].append(critic_loss.item())
            
            epoch_losses[0].append(sum(iteration_losses[0]) / len(iteration_losses[0]))
            epoch_losses[1].append(sum(iteration_losses[1]) / len(iteration_losses[1]))
        episode_actor_loss = sum(epoch_losses[0]) / len(epoch_losses[0])
        episode_critic_loss = sum(epoch_losses[1]) / len(epoch_losses[1])
        Logger.log(
            f"Actor Loss: {episode_actor_loss} Critic Loss: {episode_critic_loss} Epoch Loss: {episode_actor_loss + episode_critic_loss}",
            episode=Run.instance().dynamic_config.current_episode,
            log_type=Logger.TRAINING_TYPE,
            print_message=True)
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

    def load(self):
        experiment_path = Run.instance().experiment_path
        current_episode = Run.instance().dynamic_config.current_episode
        self.actor_state_dict = torch.load(
            f"{experiment_path}/networks/{current_episode}/actor.pth")
        self.actor = Actor()
        self.actor.load_state_dict(self.actor_state_dict)

        self.critic_state_dict = torch.load(
            f"{experiment_path}/networks/{current_episode}/critic.pth")
        self.critic = Critic()
        self.critic.load_state_dict(self.critic_state_dict)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=Run.instance().training_config.learning_rate)
        self.actor_optimizer_state_dict = torch.load(
            f"{experiment_path}/networks/{current_episode}/actor_optimizer.pth")
        self.actor_optimizer.load_state_dict(self.actor_optimizer_state_dict)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=Run.instance().training_config.learning_rate)
        self.critic_optimizer_state_dict = torch.load(
            f"{experiment_path}/networks/{current_episode}/critic_optimizer.pth")
        self.critic_optimizer.load_state_dict(self.critic_optimizer_state_dict)
