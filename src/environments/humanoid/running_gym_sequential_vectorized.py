from entities.features import Run
from entities.agents.agent import Agent
from dataclasses import dataclass
import numpy as np
import torch
from tensordict import TensorDict
from utils.logger import Logger
from torchrl.objectives.value.functional import generalized_advantage_estimate
import gymnasium as gym
from .running_gym import EnvironmentHelper as Helper


@dataclass
class Timestep:
    observation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict


class EnvironmentHelper(Helper):

    def initialize(self):
        self.environment = gym.vector.make("Humanoid-v4",
                                           render_mode="rgb_array",
                                           num_envs=self.run.environment_config.num_envs)
        self.test_environment = gym.make("Humanoid-v4", render_mode="rgb_array")
        self.timestep = Timestep(
            np.zeros((
                self.run.environment_config.num_envs,
                self.run.network_config.input_shape,
                self.run.environment_config.window_length,
            )), 0.0,
            np.zeros(self.run.environment_config.num_envs).astype(np.bool_),
            np.zeros(self.run.environment_config.num_envs).astype(np.bool_), {})
        self.test_timestep = Timestep(
            np.zeros((
                self.run.network_config.input_shape,
                self.run.environment_config.window_length,
            )), 0.0, False, False, {})

    def shift_observations(self, test_phase: bool):
        timestep: Timestep = self.get_using_environment(test_phase).timestep
        timestep.observation[..., :-1] = timestep.observation[..., 1:]

    def step(self, action: np.ndarray):
        """
        this function is called only for training phase

        Args:
            action (np.ndarray): action shape is (num_envs, action_shape)
        """
        last_timestep = Timestep(*self.environment.step(action))
        environment_timestep: Timestep = self.environment.timestep
        environment_timestep.reward = last_timestep.reward
        environment_timestep.terminated = last_timestep.terminated
        environment_timestep.truncated = last_timestep.truncated
        environment_timestep.info = last_timestep.info
        self.shift_observations(test_phase=False)
        environment_timestep.observation[:, :, -1] = last_timestep.observation
        self.rewards.append(self.timestep.reward)

    def get_state(self, test_phase: bool) -> torch.Tensor:
        timestep: Timestep = self.get_using_environment(test_phase).timestep
        next_data = torch.tensor(timestep.observation)
        if self.run.normalize_observations:
            next_data = next_data - next_data.mean()
            std = next_data.std()
            if std == 0:
                std = 1e-8
            next_data = next_data / std
        next_data = next_data.to(Run.instance().dtype)
        if test_phase:
            next_data = next_data[None, :]
        return next_data

    def reset_environment(self, test_phase):
        super().reset_environment(test_phase)
        if not test_phase:
            self.timestep.terminated = np.zeros(self.run.environment_config.num_envs).astype(
                np.bool_)
            self.timestep.truncated = np.zeros(self.run.environment_config.num_envs).astype(
                np.bool_)

    @torch.no_grad
    def rollout(self, agent: Agent):
        self.reset()
        self.reset_environment(test_phase=False)
        next_state = self.get_state(test_phase=False)
        batch_size = len(next_state)
        device = self.run.device
        for _ in range(self.run.environment_config.maximum_timesteps):
            current_state = torch.clone(next_state)
            current_state_value = agent.get_state_value(current_state)
            sub_actions, distributions = agent.act(current_state,
                                                   return_dist=True,
                                                   test_phase=False)
            action_log_prob = [
                distributions[i].log_prob(sub_actions[i]).sum() for i in range(batch_size)
            ]
            self.step(torch.cat(sub_actions, dim=0))
            next_state = self.get_state(test_phase=False)
            next_state_value = agent.get_state_value(next_state)
            memory_item = {
                'current_state': current_state.unsqueeze(1),
                'current_state_value': current_state_value.unsqueeze(1),
                'next_state_value': next_state_value.unsqueeze(1),
                'action': torch.cat(sub_actions, dim=0).unsqueeze(1),
                'action_log_prob': torch.tensor(action_log_prob).to(device)[:, None].unsqueeze(1),
                'reward': torch.tensor(self.timestep.reward)[:, None].to(device).unsqueeze(1),
                'terminated': torch.tensor(self.timestep.terminated[:,
                                                                    None]).to(device).unsqueeze(1),
                'truncated': torch.tensor(self.timestep.truncated[:, None]).to(device).unsqueeze(1)
            }
            self.memory.append(TensorDict(memory_item, batch_size=(batch_size, 1)))
        Logger.log(f"episode ended with {len(self.memory)} timesteps",
                   episode=self.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=False)
        Logger.log(f"total episode reward: {memory_item['reward'].mean()}",
                   episode=self.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        return torch.cat(self.memory, dim=1)

    @torch.no_grad
    def calculate_advantages(self, memory: TensorDict):
        rewards = memory['reward']
        if self.run.normalize_rewards:
            rewards = rewards - rewards.mean(dim=1).unsqueeze(1)
            rewards = rewards / rewards.std(dim=1).unsqueeze(1)
        terminated = memory['terminated']
        done = memory['truncated']
        done[:, -1, 0] = True
        current_state_values = memory['current_state_value']
        next_state_values = memory['next_state_value']
        advantage, value_target = generalized_advantage_estimate(Run.instance().ppo_config.gamma,
                                                                 Run.instance().ppo_config.lmbda,
                                                                 current_state_values,
                                                                 next_state_values, rewards,
                                                                 terminated, done)
        if self.run.ppo_config.normalize_advantage:
            advantage = advantage - advantage.mean(dim=1).unsqueeze(1)
            advantage = advantage / advantage.std(dim=1).unsqueeze(1)

            value_target = value_target - value_target.mean(dim=1).unsqueeze(1)
            value_target = value_target / value_target.std(dim=1).unsqueeze(1)

        memory['current_state_value_target'] = value_target
        memory['advantage'] = advantage
