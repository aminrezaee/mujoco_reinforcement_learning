from environments.helper import EnvironmentHelper as Helper
from entities.agents.agent import Agent
from abc import ABC
from tensordict import TensorDict
from utils.logger import Logger
from utils.io import add_episode_to_best_results, remove_epoch_results


class Algorithm(ABC):

    def __init__(self, environment_helper: Helper, agent: Agent):
        self.environment_helper = environment_helper
        self.agent = agent
        pass

    def test(self, visualize: bool) -> float:
        pass

    def train(self, memory: TensorDict):
        pass

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
