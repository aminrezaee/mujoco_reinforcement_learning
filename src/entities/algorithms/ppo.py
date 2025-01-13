import torch
from .base_algorithm import Algorithm
from entities.agents.agent import Agent
from entities.features import Run
from entities.timestep import Timestep
from tensordict import TensorDict
from utils.logger import Logger
from utils.io import add_episode_to_best_results, remove_epoch_results
from torchrl.objectives.value.functional import generalized_advantage_estimate
from environments.helper import EnvironmentHelper as Helper
import numpy as np
from torch.nn.functional import huber_loss, mse_loss


class PPO(Algorithm):

    @torch.no_grad
    def rollout(self):
        self.environment_helper.reset()
        self.environment_helper.reset_environment(test_phase=False)
        next_state = self.environment_helper.get_state(test_phase=False)
        batch_size = len(next_state)
        device = self.environment_helper.run.device
        for _ in range(self.environment_helper.run.environment_config.maximum_timesteps):
            current_state = torch.clone(next_state)
            current_state_value = self.agent.get_state_value(current_state)
            sub_actions, distributions = self.agent.act(current_state,
                                                        return_dist=True,
                                                        test_phase=False)
            action_log_prob = [
                distributions[i].log_prob(sub_actions[i]).sum() for i in range(batch_size)
            ]
            self.environment_helper.step(torch.cat(sub_actions, dim=0))
            next_state = self.environment_helper.get_state(test_phase=False)
            next_state_value = self.agent.get_state_value(next_state)
            memory_item = {
                'current_state':
                current_state.unsqueeze(1),
                'current_state_value':
                current_state_value.unsqueeze(1),
                'next_state_value':
                next_state_value.unsqueeze(1),
                'action':
                torch.cat(sub_actions, dim=0).unsqueeze(1),
                'action_log_prob':
                torch.tensor(action_log_prob).to(device)[:, None].unsqueeze(1),
                'reward':
                torch.tensor(self.environment_helper.timestep.reward)[:,
                                                                      None].to(device).unsqueeze(1),
                'terminated':
                torch.tensor(
                    self.environment_helper.timestep.terminated[:, None]).to(device).unsqueeze(1),
                'truncated':
                torch.tensor(self.timestep.truncated[:, None]).to(device).unsqueeze(1)
            }
            self.environment_helper.memory.append(
                TensorDict(memory_item, batch_size=(batch_size, 1)))
        Logger.log(f"episode ended with {len(self.environment_helper.memory)} timesteps",
                   episode=self.environment_helper.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=False)
        Logger.log(f"total episode reward: {memory_item['reward'].mean()}",
                   episode=self.environment_helper.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        return torch.cat(self.environment_helper.memory, dim=1)

    @torch.no_grad
    def test(self, visualize: bool):
        rewards = []
        self.environment_helper.reset_environment(test_phase=True)
        next_state = self.environment_helper.get_state(test_phase=True)
        test_timestep: Timestep = self.environment_helper.test_timestep
        while not (test_timestep.terminated or test_timestep.truncated):
            current_state = torch.clone(next_state)
            sub_actions, _ = self.agent.act(current_state, return_dist=True, test_phase=False)
            test_timestep.observation, reward, test_timestep.terminated, test_timestep.truncated, info = self.environment_helper.test_environment.step(
                torch.cat(sub_actions, dim=0).reshape(-1))
            rewards.append(reward)
            next_state = self.environment_helper.get_state(test_phase=True)
            if visualize:
                rendered_rgb_image = self.environment_helper.test_environment.render()
                self.environment_helper.images.append(rendered_rgb_image)
        if visualize:
            self.environment_helper.visualize()
        return sum(rewards) / len(rewards)

    @torch.no_grad
    def calculate_advantages(self, memory: TensorDict):
        rewards = memory['reward']
        run: Run = self.environment_helper.run
        if run.normalize_rewards:
            # rewards = rewards * 0.1
            rewards = rewards - rewards.mean(dim=1).unsqueeze(1)
        #     rewards = rewards / rewards.std(dim=1).unsqueeze(1)
        terminated = memory['terminated']
        done = memory['truncated']
        done[:, -1, 0] = True
        current_state_values = memory['current_state_value']
        next_state_values = memory['next_state_value']
        advantage, value_target = generalized_advantage_estimate(run.ppo_config.gamma,
                                                                 run.ppo_config.lmbda,
                                                                 current_state_values,
                                                                 next_state_values, rewards,
                                                                 terminated, done)
        if run.ppo_config.normalize_advantage:
            advantage = advantage - advantage.mean(dim=1).unsqueeze(1)
            # advantage = advantage / advantage.std(dim=1).unsqueeze(1)

            value_target = value_target - value_target.mean(dim=1).unsqueeze(1)
            # value_target = value_target / value_target.std(dim=1).unsqueeze(1)

        memory['current_state_value_target'] = value_target
        memory['advantage'] = advantage

    def train(self, memory: TensorDict):
        run: Run = self.environment_helper.run
        batch_size = run.training_config.batch_size
        epochs = run.training_config.epochs_per_iteration
        batches_per_epoch = int(run.environment_config.maximum_timesteps *
                                run.environment_config.num_envs / batch_size)
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
                mean, std = self.agent.actor(batch['current_state'])
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
                self.agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(),
                                               Run.instance().ppo_config.max_grad_norm)
                self.agent.critic_optimizer.step()

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
                self.agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(),
                                               Run.instance().ppo_config.max_grad_norm)
                self.agent.actor_optimizer.step()

                iteration_losses[0].append(actor_loss.detach().item())
                iteration_losses[1].append(critic_loss.detach().item())

            epoch_losses[0].append(sum(iteration_losses[0]) / len(iteration_losses[0]))
            epoch_losses[1].append(sum(iteration_losses[1]) / len(iteration_losses[1]))
        episode_actor_loss = sum(epoch_losses[0]) / len(epoch_losses[0])
        episode_critic_loss = sum(epoch_losses[1]) / len(epoch_losses[1])
        if run.dynamic_config.current_episode < 2500:
            self.agent.actor_scheduler.step()
            self.agent.critic_scheduler.step()
        Logger.log(
            f"Actor Loss: {episode_actor_loss} Critic Loss: {episode_critic_loss} Epoch Loss: {episode_actor_loss + episode_critic_loss}",
            episode=Run.instance().dynamic_config.current_episode,
            log_type=Logger.TRAINING_TYPE,
            print_message=True)
        pass

    def iterate(self):
        run = self.environment_helper.run
        Logger.log(f"-------------------------",
                   episode=run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        Logger.log(f"-------------------------",
                   episode=run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        Logger.log(f"starting iteration {run.dynamic_config.current_episode}:",
                   episode=run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        memory = self.rollout(self.agent)  # train rollout
        self.calculate_advantages(memory)
        self.train(memory)
        visualize = run.dynamic_config.current_episode % 5 == 0
        mean_rewards = self.test(visualize)  # test rollout
        self.agent.save()
        if mean_rewards > run.dynamic_config.best_reward:
            self.environment_helper.run.dynamic_config.best_reward = mean_rewards
            Logger.log(f"max reward changed to: {mean_rewards}",
                       episode=run.dynamic_config.current_episode,
                       log_type=Logger.REWARD_TYPE,
                       print_message=True)
            add_episode_to_best_results(run.experiment_path, run.dynamic_config.current_episode)
        Logger.log(f"test reward: {mean_rewards}",
                   episode=run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        run.dynamic_config.next_episode()
        removing_epoch = int(run.dynamic_config.current_episode - 10)
        remove_epoch_results(run.experiment_path, removing_epoch)
