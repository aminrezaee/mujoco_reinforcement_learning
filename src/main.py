from entities.agents.ppo_agent import PPOAgent
from environments.humanoid.running_gym_sequential_vectorized import EnvironmentHelper
import torch
from torch.nn import ELU, Tanh
from argparse import ArgumentParser
from utils.logger import Logger
from utils.io import find_experiment_name
from entities.features import *
import shutil
from os import makedirs, listdir
import os


def main():
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
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
                                         batch_size=250,
                                         epochs_per_iteration=10,
                                         minimum_learning_rate=1e-4)
        ppo_config = PPOConfig(max_grad_norm=1.0,
                               clip_epsilon=0.1,
                               gamma=0.99,
                               lmbda=0.98,
                               entropy_eps=1e-4,
                               advantage_scaler=1e+0,
                               normalize_advantage=True,
                               critic_coeffiecient=1.0)
        agent_config = AgentConfig(sub_action_count=1)
        network_config = NetworkConfig(input_shape=376,
                                       output_shape=17,
                                       output_max_value=1.0,
                                       activation_class=ELU,
                                       latent_size=128,
                                       use_bias=True)
        environment_config = EnvironmentConfig(maximum_timesteps=1000,
                                               num_envs=10,
                                               window_length=20)
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
                  dtype=torch.float32)
    Logger.log("initialize src directory!",
               episode=run.dynamic_config.current_episode,
               path=current_experiment_path,
               log_type=Logger.TRAINING_TYPE)
    run.save()
    environment_helper = EnvironmentHelper()
    agent = PPOAgent()
    if resume:
        agent.load()
        run.dynamic_config.next_episode()
    makedirs(f"{Run.instance().experiment_path}/networks/best_results", exist_ok=True)
    makedirs(f"{Run.instance().experiment_path}/visualizations/best_results", exist_ok=True)
    current_episode = Run.instance().dynamic_config.current_episode
    for i in range(current_episode, run.training_config.iteration_count):
        iterate(environment_helper, agent, run, i)


def iterate(environment_helper: EnvironmentHelper, agent: PPOAgent, run: Run, i: int):
    Logger.log(f"-------------------------",
               episode=Run.instance().dynamic_config.current_episode,
               log_type=Logger.REWARD_TYPE,
               print_message=True)
    Logger.log(f"-------------------------",
               episode=Run.instance().dynamic_config.current_episode,
               log_type=Logger.REWARD_TYPE,
               print_message=True)
    Logger.log(f"starting iteration {i}:",
               episode=Run.instance().dynamic_config.current_episode,
               log_type=Logger.REWARD_TYPE,
               print_message=True)
    memory = environment_helper.rollout(agent)  # train rollout
    environment_helper.calculate_advantages(memory)
    agent.train(memory)
    visualize = i % 5 == 0
    mean_rewards = environment_helper.test(agent, visualize)  # test rollout
    agent.save()
    if mean_rewards > run.dynamic_config.best_reward:
        run.dynamic_config.best_reward = mean_rewards
        Logger.log(f"max reward changed to: {mean_rewards}",
                   episode=Run.instance().dynamic_config.current_episode,
                   log_type=Logger.REWARD_TYPE,
                   print_message=True)
        add_episode_to_best_results()
    Logger.log(f"test reward: {mean_rewards}",
               episode=Run.instance().dynamic_config.current_episode,
               log_type=Logger.REWARD_TYPE,
               print_message=True)
    Run.instance().dynamic_config.next_episode()
    removing_epoch = int(i - 10)

    if os.path.exists(f"{run.experiment_path}/networks/{removing_epoch}"):
        shutil.rmtree(f"{run.experiment_path}/networks/{removing_epoch}")
    # if os.path.exists(f"{run.experiment_path}/visualizations/{removing_epoch}"):
    #     shutil.rmtree(f"{run.experiment_path}/visualizations/{removing_epoch}")
    if os.path.exists(f"{run.experiment_path}/debugs/{removing_epoch}"):
        shutil.rmtree(f"{run.experiment_path}/debugs/{removing_epoch}")


def add_episode_to_best_results():
    run = Run.instance()
    shutil.copytree(
        f"{run.experiment_path}/networks/{run.dynamic_config.current_episode}",
        f"{run.experiment_path}/networks/best_results/{run.dynamic_config.current_episode}")
    if os.path.exists(f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}"):
        shutil.copytree(
            f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}",
            f"{run.experiment_path}/visualizations/best_results/{run.dynamic_config.current_episode}"
        )


if __name__ == "__main__":
    main()
