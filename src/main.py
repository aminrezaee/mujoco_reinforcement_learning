from entities.agents.ppo_agent import PPOAgent
from environments.humanoid.running import EnvironmentHelper
import torch
from torch.nn import ELU
from argparse import ArgumentParser
from utils.logger import Logger
from utils.io import find_experiment_name
from entities.features import *
import shutil
from os import makedirs , listdir
import os

def main():
    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("-i", "--experiment_id", default=-1, type=int)
    parser.add_argument("-n", "--name", default="", type=str)
    args = parser.parse_args()
    experiment_id = int(args.experiment_id)
    environment_helper = EnvironmentHelper()
    max_reward = 0
    reward_config = RewardConfig()
    training_config = TrainingConfig(iteration_count=10000, learning_rate=1e-4,
                                        weight_decay=1e-4, batch_size=1024, epochs_per_iteration=1,
                                        batches_per_epoch=10, minimum_learning_rate=1e-4)
    ppo_config = PPOConfig(max_grad_norm=10.0, clip_epsilon=0.1, gamma=0.99, lmbda=0.98,
                            entropy_eps=1e-2, advantage_scaler=1e+0, normalize_advantage=True,
                            critic_coeffiecient=1.0)
    agent_config = AgentConfig()
    network_config = NetworkConfig(input_shape=67,activation_class=ELU, use_bias=False)
    environment_config = EnvironmentConfig()
    dynamic_config = DynamicConfig(0, 0, 0)
    results_dir: str = 'outputs/results'
    experiments_directory = f"{results_dir}/experiments"
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
    
    run = Run(reward_config, training_config, ppo_config, environment_config,
                agent_config, network_config, dynamic_config,
                processors=4, device='cpu', experiment_path=current_experiment_path,
                verbose=False,
                central_critic=True, central_actor=True,
                normalize_rewards=True, normalize_actions=True, normalize_observations=True,
                sequence_wise_normalization=True, dtype=torch.float32)
    Logger.log("initialize src directory!", episode=run.dynamic_config.current_episode,
               path=current_experiment_path, log_type=Logger.TRAINING_TYPE)
    agent = PPOAgent()
    makedirs(f"{Run.instance().experiment_path}/networks/best_results", exist_ok=True)
    makedirs(f"{Run.instance().experiment_path}/visualizations/best_results", exist_ok=True)
    for i in range(args.iterations):
        Logger.log(f"-------------------------" , episode=Run.instance().dynamic_config.current_episode , 
                       log_type=Logger.REWARD_TYPE, print_message=True)
        Logger.log(f"-------------------------" , episode=Run.instance().dynamic_config.current_episode , 
                       log_type=Logger.REWARD_TYPE, print_message=True)
        Logger.log(f"starting iteration {i}:" , episode=Run.instance().dynamic_config.current_episode , 
                       log_type=Logger.REWARD_TYPE, print_message=True)
        environment_helper.set_step_limit(10000)
        memory = environment_helper.rollout(agent) # train rollout
        environment_helper.calculate_advantages(memory)
        agent.train(memory)
        visualize = i % 40 == 0
        environment_helper.set_step_limit(1000)
        environment_helper.rollout(agent, visualize=visualize) # test rollout
        agent.save()
        if environment_helper.total_reward > max_reward:
            max_reward = environment_helper.total_reward
            Logger.log(f"max reward changed to: {max_reward}" , episode=Run.instance().dynamic_config.current_episode , 
                       log_type=Logger.REWARD_TYPE, print_message=True)
            add_episode_to_best_results()
        Run.instance().dynamic_config.next_episode()
        removing_epoch = int(i-10)
        
        if os.path.exists(f"{run.experiment_path}/networks/{removing_epoch}"):
            shutil.rmtree(f"{run.experiment_path}/networks/{removing_epoch}")
        if os.path.exists(f"{run.experiment_path}/visualizations/{removing_epoch}"):
            shutil.rmtree(f"{run.experiment_path}/visualizations/{removing_epoch}")
        if os.path.exists(f"{run.experiment_path}/debugs/{removing_epoch}"):
            shutil.rmtree(f"{run.experiment_path}/debugs/{removing_epoch}")
            
            
        
    

def add_episode_to_best_results():
    run = Run.instance()
    shutil.copytree(f"{run.experiment_path}/networks/{run.dynamic_config.current_episode}", f"{run.experiment_path}/networks/best_results/{run.dynamic_config.current_episode}")
    if os.path.exists(f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}"):
        shutil.copytree(f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}", 
                        f"{run.experiment_path}/visualizations/best_results/{run.dynamic_config.current_episode}")
    
if __name__ == "__main__":
    main()