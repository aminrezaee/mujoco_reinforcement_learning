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
    ovservation: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict


class EnvironmentHelper:

    def __init__(self):
        self.run = Run.instance()
        self.environment = gym.vector.make("Humanoid-v4", render_mode="rgb_array" , num_envs=self.run.environment_config.num_envs)
        self.total_reward = np.zeros(self.run.environment_config.num_envs)
        self.timestep = Timestep(np.zeros(self.run.network_config.input_shape), 0.0, False,
                                 False, {})
        self.memory = []
        self.images = []

    def reset(self):
        self.total_reward = 0
        self.memory = []
        self.images = []

    def reset_environment(self):
        self.timestep.ovservation, self.timestep.info = self.environment.reset()
        self.timestep.terminated = np.zeros(self.run.environment_config.num_envs).astype(np.bool_)
        self.timestep.truncated = np.zeros(self.run.environment_config.num_envs).astype(np.bool_)

    def step(self, action: np.ndarray):
        self.timestep = Timestep(*self.environment.step(action))
        self.total_reward += self.timestep.reward

    def get_state(self) -> torch.Tensor:
        next_data = torch.tensor(self.timestep.ovservation)
        # next_data = next_data - next_data.mean()
        # std = next_data.std()
        # if std == 0:
        #     std = 1e-8
        # next_data = next_data / std
        if len(next_data.shape) == 1:
            next_data = next_data[None , :]
        return next_data.to(self.run.dtype)

    @torch.no_grad
    def episode(self, agent: Agent, visualize: bool = False, test_phase: bool = False):
        self.reset_environment()
        next_state = self.get_state()
        sub_action_count = self.run.agent_config.sub_action_count
        batch_size = Run.instance().environment_config.num_envs
        device = self.run.device
        if any(self.timestep.truncated):
            return torch.tensor([])
        while not any(self.timestep.truncated):
            current_state = torch.clone(next_state)
            current_state_value = agent.get_state_value(current_state)
            sub_actions, distributions = agent.act(current_state,
                                                   return_dist=True,
                                                   test_phase=test_phase)
            action_log_prob = [
                distributions[i].log_prob(sub_actions[i]).sum() for i in range(batch_size)
            ]
            self.step(torch.cat(sub_actions, dim=0))
            next_state = self.get_state()
            next_state_value = agent.get_state_value(next_state)
            memory_item = {
                'current_state': current_state,
                'current_state_value': current_state_value,
                'next_state_value': next_state_value,
                'action': torch.cat(sub_actions, dim=0),
                'action_log_prob': torch.tensor(action_log_prob).to(device)[: , None],
                'reward': torch.tensor(self.timestep.reward)[:, None].to(device),
                'terminated': torch.tensor(self.timestep.terminated[:, None]).to(device),
                'truncated': torch.tensor(self.timestep.truncated[:, None]).to(device)
            }
            self.memory.append(TensorDict(memory_item, batch_size=batch_size))
            if visualize:
                rendered_rgb_image = self.environment.render()
                self.images.append(rendered_rgb_image)
        if visualize:
            self.visualize()
        Logger.log(f"episode ended with {len(self.memory)} timesteps",
                   episode=self.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=False)
        return torch.cat(self.memory, dim=0)

    def rollout(self, agent: Agent, visualize: bool = False, test_phase: bool = False):
        self.reset()
        current_rollout_size = 0
        results = []
        lengths = []
        while current_rollout_size < self.run.environment_config.maximum_timesteps:
            episode_memory = self.episode(agent, visualize, test_phase)
            current_rollout_size += len(episode_memory)
            lengths.append(len(episode_memory))
            results.append(episode_memory)
        Logger.log(f"total episode reward: {self.total_reward}",
                   episode=self.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        Logger.log(f"mean episode length: {sum(lengths)/len(lengths)}",
                   episode=self.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        return torch.cat(results, dim=0)[:self.run.environment_config.maximum_timesteps]

    def visualize(self):
        path = f"{self.run.experiment_path}/visualizations/{self.run.dynamic_config.current_episode}"
        makedirs(path, exist_ok=True)
        media.write_video(f"{path}/video.mp4", self.images, fps=30)

    @torch.no_grad
    def calculate_advantages(self, memory: TensorDict):
        rewards = memory['reward']
        terminated = memory['terminated']
        done = memory['truncated']
        done[-1] = True
        current_state_values = memory['current_state_value']
        next_state_values = memory['next_state_value']
        advantage, value_target = generalized_advantage_estimate(self.run.ppo_config.gamma,
                                                                 self.run.ppo_config.lmbda,
                                                                 current_state_values,
                                                                 next_state_values, rewards,terminated, done
                                                                 )
        advantage = advantage - advantage.mean()
        advantage = advantage / advantage.std()
        
        value_target = value_target - value_target.mean()
        value_target = value_target / value_target.std()

        memory['current_state_value_target'] = value_target
        memory['advantage'] = advantage
