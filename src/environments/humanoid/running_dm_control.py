from entities.features import Run
from entities.agents.agent import Agent
from dm_control import suite
import numpy as np
import torch
from tensordict import TensorDict
from utils.logger import Logger
from torchrl.objectives.value.functional import generalized_advantage_estimate
import mediapy as media
from os import makedirs

_STAND_HEIGHT = 1.4
_CONTROL_TIMESTEP = .025


class EnvironmentHelper:

    def __init__(self):
        self.environment = suite.load(domain_name="humanoid",
                                      task_name="run",
                                      task_kwargs={"random": 42})

        self.spec = self.environment.action_spec()
        self.timestep = self.environment.reset()
        self.total_reward = 0
        self.memory = []
        self.images = []

    def set_step_limit(self, step_limit: int):
        self.environment._step_limit = step_limit

    def reset(self):
        self.timestep = self.environment.reset()
        self.total_reward = 0
        self.memory = []
        self.images = []

    def step(self, action: np.ndarray):
        self.timestep = self.environment.step(action)
        self.total_reward += self.timestep.reward
        return self.timestep

    def get_state(self) -> torch.Tensor:
        next_data = np.array([])
        for key, value in self.timestep.observation.items():
            # print(f"{key}: {value}")
            next_data = np.append(next_data, value)
        next_data = torch.tensor(next_data, dtype=torch.float32)
        next_data = next_data - next_data.mean()
        std = next_data.std()
        if std == 0:
            std = 1e-8
        next_data = next_data / std
        return next_data[None, :]

    @torch.no_grad
    def rollout(self, agent: Agent, visualize: bool = False, test_phase: bool = False):
        self.reset()
        next_state = self.get_state()
        sub_action_count = Run.instance().agent_config.sub_action_count
        while not self.timestep.last():
            current_state = torch.clone(next_state)
            current_state_value = agent.get_state_value(current_state)
            sub_actions, distributions = agent.act(current_state,
                                                   return_dist=True,
                                                   test_phase=test_phase)
            action_log_prob = [
                distributions[i].log_prob(sub_actions[i]).sum() for i in range(sub_action_count)
            ]
            self.timestep = self.step(torch.cat(sub_actions, dim=1))
            next_state = self.get_state()
            next_state_value = agent.get_state_value(next_state)
            memory_item = {
                'current_state': current_state,
                'current_state_value': current_state_value,
                'next_state_value': next_state_value,
                'action': torch.cat(sub_actions, dim=0)[None, :],
                'action_log_prob': torch.tensor(action_log_prob).to(current_state.device)[None, :],
                'reward': torch.tensor([[self.timestep.reward]]).to(current_state.device)
            }
            self.memory.append(TensorDict(memory_item, batch_size=1))
            if visualize:
                rendered_rgb_image = self.environment.physics.render(height=160, width=240)
                self.images.append(rendered_rgb_image)
        Logger.log(f"total episode reward: {self.total_reward}",
                   episode=Run.instance().dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        if visualize:
            self.visualize()
        return torch.cat(self.memory, dim=0)

    def visualize(self):
        run = Run.instance()
        path = f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}"
        makedirs(path, exist_ok=True)
        media.write_video(f"{path}/video.mp4", self.images, fps=30)

    @torch.no_grad
    def calculate_advantages(self, memory: TensorDict):
        rewards = memory['reward']
        # rewards = rewards - rewards.mean()
        # rewards = rewards/rewards.std()
        done = torch.zeros_like(rewards)
        done[-1] = 1
        done = done.to(torch.bool)
        current_state_values = memory['current_state_value']
        next_state_values = memory['next_state_value']
        advantage, value_target = generalized_advantage_estimate(Run.instance().ppo_config.gamma,
                                                                 Run.instance().ppo_config.lmbda,
                                                                 current_state_values,
                                                                 next_state_values, rewards, done,
                                                                 done)

        memory['current_state_value_target'] = value_target
        memory['advantage'] = advantage
