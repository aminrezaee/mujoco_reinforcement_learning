"""" this class is used for training features and environment features"""
from dataclasses import asdict, dataclass
from typing import List

import torch

from utils.io import _get_dict_name, read_json_config
from utils.type_utils import Singleton


@dataclass
class RewardConfig:
    pass


@dataclass
class TrainingConfig:
    iteration_count: int  # Number of training iterations
    learning_rate: float  # Learning rate
    weight_decay: float  # coefficient norm of weights term in loss
    batch_size: float  # for example 0.25 means each batch has size int(0.25 * maximum_timesteps)
    epochs_per_iteration: int  # Number of optimization steps per training iteration
    minimum_learning_rate: float
    agents_dir: str = './outputs/agents'
    save_per_iteration: int = 10


@dataclass
class EnvironmentConfig:
    maximum_timesteps: int
    num_envs:int


@dataclass
class AgentConfig:
    sub_action_count: int


@dataclass
class NetworkConfig:
    input_shape: int
    output_shape: int
    output_max_value: float
    activation_class: torch.nn.Module
    use_bias: bool


@dataclass
class DynamicConfig:
    current_episode: int
    current_episode_timestep: int
    current_timestep: int

    def next_episode(self):
        self.current_episode = int(self.current_episode + 1)

    def next_timestep(self):
        self.current_episode_timestep = int(self.current_episode_timestep + 1)
        self.current_timestep = int(self.current_timestep + 1)

    def reset_timestep(self):
        self.current_episode_timestep = 0

    def set_episode(self, episode: int):
        self.current_episode = episode


@dataclass
class PPOConfig:
    max_grad_norm: float  # Maximum norm for the gradients
    clip_epsilon: float  # clip value for PPO loss
    gamma: float  # discount factor
    lmbda: float  # lambda for generalised advantage estimation
    entropy_eps: float  # coefficient of the entropy term in the PPO loss
    advantage_scaler: float
    normalize_advantage: bool
    critic_coeffiecient: float


@dataclass
class Run(metaclass=Singleton):
    rewards_config: RewardConfig
    training_config: TrainingConfig
    ppo_config: PPOConfig
    environment_config: EnvironmentConfig
    agent_config: AgentConfig
    network_config: NetworkConfig
    dynamic_config: DynamicConfig
    processors: int
    device: str
    experiment_path: str
    verbose: bool
    central_critic: bool
    central_actor: bool
    normalize_rewards: bool
    normalize_actions: bool
    normalize_observations: bool
    sequence_wise_normalization: bool
    dtype: torch.dtype

    def get_name(self):
        dictionary = asdict(self)
        return _get_dict_name(dictionary)

    @staticmethod
    def instance() -> 'Run':
        if len(list(Run._instances.values())):
            return list(Run._instances.values())[0]
        return None


def get_configurations(experiment_path: str) -> Run:
    configuration_file_path = f"{experiment_path}/configurations.json"
    configurations = read_json_config(configuration_file_path)['run']
    configurations['dtype'] = getattr(torch, configurations['dtype'])
    rewards_config = configurations.pop("rewards_config").values()
    training_config = configurations.pop('training_config').values()
    configurations['network_config']['activation_class'] = getattr(
        torch.nn, configurations['network_config']['activation_class'])
    network_config = configurations.pop('network_config').values()
    ppo_config = configurations.pop('ppo_config').values()
    environment_config = configurations.pop('environment_config').values()
    agent_config = configurations.pop('agent_config').values()
    dynamic_config = configurations.pop('dynamic_config').values()
    run = Run(RewardConfig(*rewards_config), TrainingConfig(*training_config),
              PPOConfig(*ppo_config), EnvironmentConfig(*environment_config),
              AgentConfig(*agent_config), NetworkConfig(*network_config),
              DynamicConfig(*dynamic_config), *configurations.values())
    return run
