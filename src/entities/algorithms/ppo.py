import torch
from .base_algorithm import Algorithm
from entities.features import Run
from tensordict import TensorDict
from utils.logger import Logger
from torchrl.objectives.value.functional import generalized_advantage_estimate
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
            action_log_prob = distributions.log_prob(sub_actions).sum(dim=1)
            self.environment_helper.step(sub_actions)
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
                sub_actions.unsqueeze(1),
                'action_log_prob':
                action_log_prob.to(device).unsqueeze(1),
                'reward':
                torch.tensor(self.environment_helper.timestep.reward)[:,
                                                                      None].to(device).unsqueeze(1),
                'terminated':
                torch.tensor(self.environment_helper.timestep.terminated[:, None]).to(device),
                'truncated':
                torch.tensor(self.environment_helper.timestep.truncated[:, None]).to(device)
            }
            self.environment_helper.memory.append(
                TensorDict(memory_item, batch_size=(batch_size, 1)))
        Logger.log(f"episode ended with {len(self.environment_helper.memory)} timesteps",
                   episode=self.environment_helper.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=False)
        Logger.log(
            f"total episode reward: {torch.cat(self.environment_helper.memory, dim=1)['reward'].mean()}",
            episode=self.environment_helper.run.dynamic_config.current_episode,
            log_type=Logger.REWARD_TYPE,
            print_message=True)
        return torch.cat(self.environment_helper.memory, dim=1)

    @torch.no_grad
    def calculate_advantages(self, memory: TensorDict):
        rewards = memory['reward']
        run: Run = self.environment_helper.run
        if run.normalize_rewards:
            # rewards = rewards * 0.1
            rewards = rewards - rewards.mean(dim=1).unsqueeze(1)
        #     rewards = rewards / rewards.std(dim=1).unsqueeze(1)
        terminated = memory['terminated'].unsqueeze(-1)
        done = torch.clone(terminated)
        done[:, -1, :] = True  # in last timestep the trajectory ended.

        current_state_values = memory['current_state_value']
        next_state_values = memory['next_state_value']
        advantage, value_target = generalized_advantage_estimate(run.ppo_config.gamma,
                                                                 run.ppo_config.lmbda,
                                                                 current_state_values,
                                                                 next_state_values, rewards, done,
                                                                 terminated)
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
                _, distributions = self.agent.act(batch['current_state'], return_dist=True)
                action_log_prob = batch['action_log_prob'][:, joint_index]

                new_action_log_prob = distributions.log_prob(sub_actions).sum()[None]
                # critic loss
                current_state_value = self.agent.get_state_value(batch['current_state'])
                current_state_value_target = batch['current_state_value_target']
                critic_loss: torch.Tensor = mse_loss(current_state_value,
                                                     current_state_value_target,
                                                     reduction='mean')
                self.agent.optimizers['critic'].zero_grad()
                critic_loss.backward()
                self.agent.optimizers['critic'].step()
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
                self.agent.optimizers['critic'].zero_grad()
                actor_loss.backward()
                self.agent.optimizers['critic'].step()
                torch.nn.utils.clip_grad_norm_(self.agent.networks.parameters(),
                                               Run.instance().ppo_config.max_grad_norm)

                iteration_losses[0].append(actor_loss.detach().item())
                iteration_losses[1].append(critic_loss.detach().item())

            epoch_losses[0].append(sum(iteration_losses[0]) / len(iteration_losses[0]))
            epoch_losses[1].append(sum(iteration_losses[1]) / len(iteration_losses[1]))
        episode_actor_loss = sum(epoch_losses[0]) / len(epoch_losses[0])
        episode_critic_loss = sum(epoch_losses[1]) / len(epoch_losses[1])
        if run.dynamic_config.current_episode < 2500:
            for scheduler in self.agent.schedulers.values():
                scheduler.step()
        Logger.log(
            f"Actor Loss: {episode_actor_loss} Critic Loss: {episode_critic_loss} Epoch Loss: {episode_actor_loss + episode_critic_loss}",
            episode=Run.instance().dynamic_config.current_episode,
            log_type=Logger.TRAINING_TYPE,
            print_message=True)
        pass

    def _iterate(self):
        memory = self.rollout()  # train rollout
        self.calculate_advantages(memory)
        self.train(memory)
