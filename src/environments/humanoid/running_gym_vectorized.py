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
        self.test_environment = gym.make("Humanoid-v4" , render_mode="rgb_array")
        self.total_reward = np.zeros(self.run.environment_config.num_envs)
        self.timestep = Timestep(np.zeros(self.run.network_config.input_shape), 0.0, False,
                                 False, {})
        self.memory = []
        self.images = []

    def reset(self):
        self.total_reward = np.zeros(self.run.environment_config.num_envs)
        self.memory = []
        self.images = []
        
    def get_using_environment(self , test_phase:bool):
        using_environment = self.environment
        if test_phase:
            using_environment = self.test_environment
        return using_environment

    def reset_environment(self , test_phase:bool):
        using_environment = self.get_using_environment(test_phase)
        self.timestep.ovservation, self.timestep.info = using_environment.reset()
        self.timestep.terminated = np.zeros(self.run.environment_config.num_envs).astype(np.bool_)
        self.timestep.truncated = np.zeros(self.run.environment_config.num_envs).astype(np.bool_)

    def step(self, action: np.ndarray , test_phase:bool):
        using_environment = self.get_using_environment(test_phase)
        self.timestep = Timestep(*using_environment.step(action))
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
    def test(self , agent:Agent , visualize:bool):
        rewards = []
        self.reset_environment(test_phase = True)
        next_state = self.get_state()
        for _ in range(self.run.environment_config.maximum_timesteps):
            current_state = torch.clone(next_state)
            sub_actions, _ = agent.act(current_state,
                                                   return_dist=True,
                                                   test_phase=False)
            ovservation, reward , terminated , truncated , info = self.test_environment.step(torch.cat(sub_actions, dim=0))
            rewards.append(reward)
            if visualize:
                rendered_rgb_image = self.test_environment.render()
                self.images.append(rendered_rgb_image)
        if visualize:
            self.visualize()
        return sum(rewards)/len(rewards)


    @torch.no_grad
    def rollout(self, agent: Agent):
        self.reset_environment(test_phase = False)
        next_state = self.get_state()
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
            self.step(torch.cat(sub_actions, dim=0), False)
            next_state = self.get_state()
            next_state_value = agent.get_state_value(next_state)
            memory_item = {
                'current_state': current_state.unsqueeze(-2),
                'current_state_value': current_state_value.unsqueeze(-2),
                'next_state_value': next_state_value.unsqueeze(-2),
                'action': torch.cat(sub_actions, dim=0).unsqueeze(-2),
                'action_log_prob': torch.tensor(action_log_prob).to(device)[: , None].unsqueeze(-2),
                'reward': torch.tensor(self.timestep.reward)[:, None].to(device).unsqueeze(-2),
                'terminated': torch.tensor(self.timestep.terminated[:, None]).to(device).unsqueeze(-2),
                'truncated': torch.tensor(self.timestep.truncated[:, None]).to(device).unsqueeze(-2)
            }
            self.memory.append(TensorDict(memory_item, batch_size=(batch_size,1)))
        Logger.log(f"episode ended with {len(self.memory)} timesteps",
                   episode=self.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=False)
        Logger.log(f"total episode reward: {memory_item['reward'].mean()}",
                   episode=self.run.dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        return torch.cat(self.memory, dim=1)

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
        advantage = advantage - advantage.mean(dim=1).unsqueeze(1)
        advantage = advantage / advantage.std(dim=1).unsqueeze(1)
        
        value_target = value_target - value_target.mean(dim=1).unsqueeze(1)
        value_target = value_target / value_target.std(dim=1).unsqueeze(1)

        memory['current_state_value_target'] = value_target
        memory['advantage'] = advantage
