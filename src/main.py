from entities.agents.ppo_agent import PPOAgent
from entities.agents.soft_actor_critic_agent import SoftActorCriticAgent
from environments.humanoid.running_gym_sequential_vectorized import EnvironmentHelper
import torch
from torch.nn import ReLU, ELU
from argparse import ArgumentParser
from utils.logger import Logger
from utils.io import find_experiment_name
from entities.features import *
from entities.algorithms.ppo import PPO
from entities.algorithms.soft_actor_critic import SoftActorCritic
from os import makedirs, listdir
import mlflow


def main():
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.pytorch.autolog(log_every_n_epoch=1)
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("-i", "--experiment_id", default=-1, type=int)
    parser.add_argument("-it", "--resume_iteration", default=-1, type=int)
    parser.add_argument("-n", "--name", default="", type=str)
    args = parser.parse_args()
    results_dir: str = 'outputs/results'
    experiments_directory = f"{results_dir}/experiments"
    experiment_id = int(args.experiment_id)
    if args.resume_iteration > 0:
        # resume the experiment
        resume = True
        experiment_name = find_experiment_name(experiment_id, experiments_directory)
        current_experiment_path = f"{experiments_directory}/{experiment_id}_{experiment_name}"
        run = Run.get_configurations(current_experiment_path)
        run.dynamic_config.current_episode = int(args.resume_iteration)
    else:
        reward_config = RewardConfig()
        training_config = TrainingConfig(iteration_count=args.iterations,
                                         learning_rate=1e-4,
                                         weight_decay=1e-4,
                                         batch_size=500,
                                         epochs_per_iteration=1,
                                         minimum_learning_rate=1e-4)
        ppo_config = PPOConfig(max_grad_norm=1.0,
                               clip_epsilon=0.1,
                               gamma=0.99,
                               lmbda=0.98,
                               entropy_eps=1e-4,
                               advantage_scaler=1e+0,
                               normalize_advantage=True,
                               critic_coeffiecient=1.0)
        sac_config = SACConfig(max_grad_norm=1.0,
                               gamma=0.99,
                               alpha=0.05,
                               tau=0.005,
                               memory_capacity=999,
                               target_update_interval=1,
                               automatic_entropy_tuning=False)
        agent_config = AgentConfig(sub_action_count=1)
        network_config = NetworkConfig(input_shape=348,
                                       output_shape=17,
                                       output_max_value=1.0,
                                       activation_class=ReLU,
                                       num_linear_layers=4,
                                       linear_hidden_shapes=[256, 256, 128, 128],
                                       num_lstm_layers=1,
                                       lstm_latent_size=256,
                                       use_bias=True,
                                       use_batch_norm=False,
                                       feature_extractor="LSTM")
        environment_config = EnvironmentConfig(maximum_timesteps=500, num_envs=5, window_length=5)
        dynamic_config = DynamicConfig(0, 0, 0, 0)
        makedirs(experiments_directory, exist_ok=True)
        if experiment_id < 0:  # then create a new one
            directories = listdir(experiments_directory)
        experiment_id = 0 if len(directories) == 0 else int(
            max([int(item.split("_")[0]) for item in directories]) + 1)
        resume = False
        if len(args.name) == 0:
            experiment_name = find_experiment_name(experiment_id, experiments_directory)
            current_experiment_path = f"{experiments_directory}/{experiment_id}_{experiment_name}"
        else:
            current_experiment_path = f"{experiments_directory}/{experiment_id}_{args.name}"

        run = Run(reward_config,
                  training_config,
                  ppo_config,
                  sac_config,
                  environment_config,
                  agent_config,
                  network_config,
                  dynamic_config,
                  processors=4,
                  device='cpu',
                  experiment_path=current_experiment_path,
                  verbose=False,
                  central_critic=True,
                  central_actor=True,
                  normalize_rewards=True,
                  normalize_actions=True,
                  normalize_observations=True,
                  sequence_wise_normalization=True,
                  dtype=torch.float32,
                  render_size=[200, 200])
    Logger.log("initialize src directory!",
               episode=run.dynamic_config.current_episode,
               path=current_experiment_path,
               log_type=Logger.TRAINING_TYPE)
    run.save()
    environment_helper = EnvironmentHelper()
    modules = dict()

    agent = SoftActorCriticAgent()
    if resume:
        agent.load()
        run.dynamic_config.next_episode()
    makedirs(f"{Run.instance().experiment_path}/networks/best_results", exist_ok=True)
    makedirs(f"{Run.instance().experiment_path}/visualizations/best_results", exist_ok=True)
    current_episode = Run.instance().dynamic_config.current_episode
    algorithm = SoftActorCritic(environment_helper, agent)
    run_tags = {
        "lr": f"{run.training_config.learning_rate}",
        "batch_size": f"{run.training_config.batch_size}",
        "num_envs": f"{run.environment_config.num_envs}",
        "window_length": f"{run.environment_config.window_length}",
        "feature_extractor": run.network_config.feature_extractor,
        "algorithm": algorithm.__class__.__name__,
        "net_activation": run.network_config.activation_class.__name__,
        "normalize_observations": f"{run.normalize_observations}",
        "normalize_rewards": f"{run.normalize_rewards}",
        "num_lstm_layers": f"{run.network_config.num_lstm_layers}",
        "lstm_latent_size": f"{run.network_config.lstm_latent_size}",
        "num_linear_layers": f"{run.network_config.num_linear_layers}",
    }
    with mlflow.start_run(tags=run_tags) as mlflow_run:
        for i in range(current_episode, run.training_config.iteration_count):
            algorithm.iterate()


if __name__ == "__main__":
    main()
