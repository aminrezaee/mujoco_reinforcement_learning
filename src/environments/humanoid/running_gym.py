from entities.features import Run
from entities.agents.agent import Agent
from dataclasses import dataclass
import numpy as np
import torch
from tensordict import TensorDict
from utils.logger import Logger
from torchrl.objectives.value.functional import generalized_advantage_estimate
import mediapy as media
from os import makedirs
import gymnasium as gym


@dataclass
class Timestep:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class EnvironmentHelper:

    def __init__(self):
        self.run = Run.instance()
        self.environment = gym.make("Humanoid-v4", render_mode="rgb_array")
        self.test_environment = gym.make("Humanoid-v4", render_mode="rgb_array")
        self.rewards = []
        self.timestep = Timestep(np.zeros(Run.instance().network_config.input_shape), 0.0, False,
                                 False, {})
        self.memory = []
        self.images = []

    def reset(self):
        self.rewards = []
        self.memory = []
        self.images = []

    def get_using_environment(self, test_phase: bool):
        using_environment = self.environment
        if test_phase:
            using_environment = self.test_environment
        return using_environment

    def reset_environment(self, test_phase: bool):
        using_environment = self.get_using_environment(test_phase)
        self.timestep.observation, self.timestep.info = using_environment.reset()
        self.timestep.terminated = False
        self.timestep.truncated = False

    @torch.no_grad
    def test(self, agent: Agent, visualize: bool):
        rewards = []
        self.reset_environment(test_phase=True)
        next_state = self.get_state()
        while not (self.timestep.terminated or self.timestep.truncated):
            current_state = torch.clone(next_state)
            sub_actions, _ = agent.act(current_state, return_dist=True, test_phase=False)
            self.timestep.observation, reward, self.timestep.terminated, self.timestep.truncated, info = self.test_environment.step(
                torch.cat(sub_actions, dim=0).reshape(-1))
            rewards.append(reward)
            next_state = self.get_state()
            if visualize:
                rendered_rgb_image = self.test_environment.render()
                self.images.append(rendered_rgb_image)
        if visualize:
            self.visualize()
        return sum(rewards) / len(rewards)

    def step(self, action: np.ndarray, test_phase: bool):
        using_environment = self.get_using_environment(test_phase)
        self.timestep = Timestep(*using_environment.step(action))
        self.rewards.append(self.timestep.reward)

    def get_state(self) -> torch.Tensor:
        next_data = torch.tensor(self.timestep.observation)
        if self.run.normalize_observations:
            next_data = next_data - next_data.mean()
            std = next_data.std()
            if std == 0:
                std = 1e-8
            next_data = next_data / std
        return next_data[None, :].to(Run.instance().dtype)

    @torch.no_grad
    def episode(self, agent: Agent, visualize: bool = False, test_phase: bool = False):
        self.reset_environment(test_phase)
        next_state = self.get_state()
        sub_action_count = Run.instance().agent_config.sub_action_count
        device = Run.instance().device
        if self.timestep.terminated or self.timestep.truncated:
            return torch.tensor([])
        while not (self.timestep.terminated or self.timestep.truncated):
            current_state = torch.clone(next_state)
            current_state_value = agent.get_state_value(current_state)
            sub_actions, distributions = agent.act(current_state,
                                                   return_dist=True,
                                                   test_phase=test_phase)
            action_log_prob = [
                distributions[i].log_prob(sub_actions[i]).sum() for i in range(sub_action_count)
            ]
            self.step(torch.cat(sub_actions, dim=1).reshape(-1).cpu().numpy(), test_phase)
            next_state = self.get_state()
            next_state_value = agent.get_state_value(next_state)
            memory_item = {
                'current_state': current_state,
                'current_state_value': current_state_value,
                'next_state_value': next_state_value,
                'action': torch.cat(sub_actions, dim=0)[None, :],
                'action_log_prob': torch.tensor(action_log_prob).to(device)[None, :],
                'reward': torch.tensor([[self.timestep.reward]]).to(device),
                'terminated': torch.tensor([[self.timestep.terminated]]).to(device),
                'truncated': torch.tensor([[self.timestep.truncated]]).to(device)
            }
            self.memory.append(TensorDict(memory_item, batch_size=1))
            if visualize:
                rendered_rgb_image = self.environment.render()
                self.images.append(rendered_rgb_image)
        if visualize:
            self.visualize()
        Logger.log(f"episode ended with {len(self.memory)} timesteps",
                   episode=Run.instance().dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=False)
        return torch.cat(self.memory, dim=0)

    def rollout(self, agent: Agent, visualize: bool = False, test_phase: bool = False):
        self.reset()
        run = Run.instance()
        current_rollout_size = 0
        results = []
        lengths = []
        while current_rollout_size < run.environment_config.maximum_timesteps:
            episode_memory = self.episode(agent, visualize, test_phase)
            current_rollout_size += len(episode_memory)
            lengths.append(len(episode_memory))
            results.append(episode_memory)
        Logger.log(f"total episode reward: {sum(self.rewards)/len(self.rewards)}",
                   episode=Run.instance().dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        Logger.log(f"mean episode length: {sum(lengths)/len(lengths)}",
                   episode=Run.instance().dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        return torch.cat(results, dim=0)[:run.environment_config.maximum_timesteps]

    def visualize(self):
        run = Run.instance()
        path = f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}"
        makedirs(path, exist_ok=True)
        media.write_video(f"{path}/video.mp4", self.images, fps=30)

    @torch.no_grad
    def calculate_advantages(self, memory: TensorDict):
        rewards = memory['reward']
        if self.run.normalize_rewards:
            rewards = rewards - rewards.mean()
            rewards = rewards / rewards.std()
        terminated = memory['terminated']
        done = memory['truncated']
        done[-1] = True
        current_state_values = memory['current_state_value']
        next_state_values = memory['next_state_value']
        advantage, value_target = generalized_advantage_estimate(Run.instance().ppo_config.gamma,
                                                                 Run.instance().ppo_config.lmbda,
                                                                 current_state_values,
                                                                 next_state_values, rewards,
                                                                 terminated, done)
        if self.run.ppo_config.normalize_advantage:
            advantage = advantage - advantage.mean()
            advantage = advantage / advantage.std()

            value_target = value_target - value_target.mean()
            value_target = value_target / value_target.std()

        memory['current_state_value_target'] = value_target
        memory['advantage'] = advantage
