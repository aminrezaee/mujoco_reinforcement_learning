import torch
from .base_algorithm import Algorithm
from entities.agents.agent import Agent
from entities.features import Run
from entities.timestep import Timestep
from tensordict import TensorDict
from utils.logger import Logger
from torchrl.objectives.value.functional import generalized_advantage_estimate
from gymnasium.core import Env


class PPO(Algorithm):

    def __init__(self, environment: Env):

        super().__init__(environment)

    @torch.no_grad
    def rollout(self, agent: Agent):
        self.environment_helper.reset()
        self.environment_helper.reset_environment(test_phase=False)
        next_state = self.environment_helper.get_state(test_phase=False)
        batch_size = len(next_state)
        device = self.environment_helper.run.device
        for _ in range(self.environment_helper.run.environment_config.maximum_timesteps):
            current_state = torch.clone(next_state)
            current_state_value = agent.get_state_value(current_state)
            sub_actions, distributions = agent.act(current_state,
                                                   return_dist=True,
                                                   test_phase=False)
            action_log_prob = [
                distributions[i].log_prob(sub_actions[i]).sum() for i in range(batch_size)
            ]
            self.environment_helper.step(torch.cat(sub_actions, dim=0))
            next_state = self.environment_helper.get_state(test_phase=False)
            next_state_value = agent.get_state_value(next_state)
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
    def test(self, agent: Agent, visualize: bool):
        rewards = []
        self.environment_helper.reset_environment(test_phase=True)
        next_state = self.environment_helper.get_state(test_phase=True)
        test_timestep: Timestep = self.environment_helper.test_timestep
        while not (test_timestep.terminated or test_timestep.truncated):
            current_state = torch.clone(next_state)
            sub_actions, _ = agent.act(current_state, return_dist=True, test_phase=False)
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
