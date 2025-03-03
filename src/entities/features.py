"""" this class is used for training features and environment features"""
from dataclasses import asdict, dataclass
from typing import List

import torch

from utils.io import _get_dict_name, read_json_config
from utils.type_utils import Singleton
import json


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
    num_envs: int
    window_length: int


@dataclass
class AgentConfig:
    sub_action_count: int


@dataclass
class NetworkConfig:
    input_shape: int
    output_shape: int
    output_max_value: float
    activation_class: torch.nn.Module
    num_linear_layers: int
    linear_hidden_shapes: list[int]
    num_feature_extractor_layers: int
    feature_extractor_latent_size: int
    use_bias: bool
    use_batch_norm: bool
    feature_extractor: str
    last_layer_std: float


@dataclass
class DynamicConfig:
    current_episode: int
    current_episode_timestep: int
    current_timestep: int
    best_reward: float

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
class SACConfig:
    max_grad_norm: float  # Maximum norm for the gradients
    gamma: float  # discount factor
    alpha: float
    tau: float
    memory_capacity: int
    target_update_interval: int
    automatic_entropy_tuning: bool


@dataclass
class Run(metaclass=Singleton):
    rewards_config: RewardConfig
    training_config: TrainingConfig
    ppo_config: PPOConfig
    sac_config: SACConfig
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
    render_size: List[int]

    def get_name(self):
        dictionary = asdict(self)
        return _get_dict_name(dictionary)

    @staticmethod
    def instance() -> 'Run':
        if len(list(Run._instances.values())):
            return list(Run._instances.values())[0]
        return None

    def save(self):
        configurations = {'run': asdict(self)}
        configurations["run"]['dtype'] = str(configurations["run"]['dtype']).split(".")[-1]
        configurations["run"]['network_config']['activation_class'] = configurations["run"][
            'network_config']['activation_class'].__name__
        configurations = json.dumps(configurations, indent=4)
        with open(f"{self.experiment_path}/configurations.json", "w") as configurations_file:
            configurations_file.writelines(configurations)
            configurations_file.flush()
            configurations_file.close()

    @staticmethod
    def get_configurations(experiment_path: str) -> 'Run':
        configuration_file_path = f"{experiment_path}/configurations.json"
        configurations = read_json_config(configuration_file_path)['run']
        configurations['dtype'] = getattr(torch, configurations['dtype'])
        rewards_config = configurations.pop("rewards_config").values()
        training_config = configurations.pop('training_config').values()
        configurations['network_config']['activation_class'] = getattr(
            torch.nn, configurations['network_config']['activation_class'])
        network_config = configurations.pop('network_config').values()
        ppo_config = configurations.pop('ppo_config').values()
        sac_config = configurations.pop('sac_config').values()
        environment_config = configurations.pop('environment_config').values()
        agent_config = configurations.pop('agent_config').values()
        dynamic_config = configurations.pop('dynamic_config').values()
        run = Run(RewardConfig(*rewards_config), TrainingConfig(*training_config),
                  PPOConfig(*ppo_config), SACConfig(*sac_config),
                  EnvironmentConfig(*environment_config), AgentConfig(*agent_config),
                  NetworkConfig(*network_config), DynamicConfig(*dynamic_config),
                  *configurations.values())
        return run
