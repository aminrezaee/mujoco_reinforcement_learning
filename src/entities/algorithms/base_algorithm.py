from environments.helper import EnvironmentHelper as Helper
from entities.agents.agent import Agent
from abc import ABC
from tensordict import TensorDict
from utils.logger import Logger
from utils.io import add_episode_to_best_results, remove_epoch_results
from utils.error_handling_utils import timeit
import torch
from entities.timestep import Timestep


class Algorithm(ABC):

    def __init__(self, environment_helper: Helper, agent: Agent):
        self.environment_helper = environment_helper
        self.agent = agent
        pass

    @torch.no_grad
    def test(self, visualize: bool):
        self.agent.networks.eval()
        rewards = []
        self.environment_helper.reset_environment(test_phase=True)
        next_state = self.environment_helper.get_state(test_phase=True)
        test_timestep: Timestep = self.environment_helper.test_timestep
        for _ in range(1000):
            current_state = torch.clone(next_state)
            action, _ = self.agent.act(current_state, return_dist=True, test_phase=False)
            last_observation, reward, test_timestep.terminated, test_timestep.truncated, info = self.environment_helper.test_environment.step(
                action.reshape(-1))
            if test_timestep.terminated:
                self.environment_helper.reset_environment(test_phase=True)
            else:
                self.environment_helper.shift_observations(test_phase=True, environment_index=-1)
                test_timestep.observation[:, -1] = last_observation
            rewards.append(reward)
            next_state = self.environment_helper.get_state(test_phase=True)
            if visualize:
                rendered_rgb_image = self.environment_helper.test_environment.render()
                self.environment_helper.images.append(rendered_rgb_image)
        if visualize:
            self.environment_helper.visualize()
        self.agent.networks.train()
        return sum(rewards) / len(rewards)

    def train(self, memory: TensorDict):
        pass

    @timeit
    def iterate(self):
        self.__log_start_of_iteration()
        self._iterate()
        self.__save_iteration_results()
        pass

    def _iterate(self):
        pass

    def __save_iteration_results(self):
        run = self.environment_helper.run
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

    def __log_start_of_iteration(self):
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
