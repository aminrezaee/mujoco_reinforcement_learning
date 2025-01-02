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
from dm_control.utils import rewards
from dm_control.suite.humanoid import Physics
from dm_control.suite import common
from dm_control.rl import control
def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model('humanoid.xml'), common.ASSETS

_STAND_HEIGHT = 1.4
_CONTROL_TIMESTEP = .025
class Environment(suite.humanoid.Humanoid):
    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.head_height(),
                                    bounds=(_STAND_HEIGHT, float('inf')),
                                    margin=_STAND_HEIGHT/4)
        return standing
        # upright = rewards.tolerance(physics.torso_upright(),
        #                             bounds=(0.9, float('inf')), sigmoid='linear',
        #                             margin=1.9, value_at_margin=0)
        # stand_reward = standing * upright
        # small_control = rewards.tolerance(physics.control(), margin=1,
        #                                 value_at_margin=0,
        #                                 sigmoid='quadratic').mean()
        # small_control = (4 + small_control) / 5
        # if self._move_speed == 0:
        #     horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
        #     dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
        #     return small_control * stand_reward * dont_move
        # else:
        #     com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
        #     move = rewards.tolerance(com_velocity,
        #                             bounds=(self._move_speed, float('inf')),
        #                             margin=self._move_speed, value_at_margin=0,
        #                             sigmoid='linear')
        #     move = (5*move + 1) / 6
        #     return small_control * stand_reward * move

class EnvironmentHelper:
    def __init__(self):
        physics = Physics.from_xml_string(*get_model_and_assets())
        task = Environment(move_speed=10, pure_state=False, random=42)
        environment_kwargs = {}
        self.environment = control.Environment(
            physics, task, time_limit=1000, control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
        self.spec = self.environment.action_spec()
        self.timestep = self.environment.reset()
        self.total_reward = 0
        self.memory = []
        self.images = []
        
    def set_step_limit(self , step_limit:int):
        self.environment._step_limit = step_limit
        
    def reset(self):
        self.timestep = self.environment.reset()
        self.total_reward = 0
        self.memory = []
        self.images = []
        
    def step(self, action:np.ndarray):
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
        if std ==0: 
            std = 1e-8
        next_data = next_data/std
        return next_data[None,:]
    
    @torch.no_grad
    def rollout(self , agent:Agent , visualize:bool=False , test_phase:bool=False):
        self.reset()
        next_state = self.get_state()
        sub_action_count = Run.instance().agent_config.sub_action_count
        while not self.timestep.last():
            current_state = torch.clone(next_state)
            current_state_value = agent.get_state_value(current_state)
            sub_actions , distributions  = agent.act(current_state , return_dist=True , test_phase=test_phase)
            action_log_prob = [distributions[i].log_prob(sub_actions[i]).sum() for i in range(sub_action_count)]
            self.timestep = self.step(torch.cat(sub_actions , dim=1))
            next_state = self.get_state()
            next_state_value = agent.get_state_value(next_state)
            memory_item = {'current_state':current_state,
                           'current_state_value':current_state_value , 
                           'next_state_value':next_state_value, 
                           'action':torch.cat(sub_actions , dim=0)[None,:],
                           'action_log_prob':torch.tensor(action_log_prob).to(current_state.device)[None,:],
                           'reward':torch.tensor([[self.timestep.reward]]).to(current_state.device)}
            self.memory.append(TensorDict(memory_item,batch_size=1))
            if visualize:
                rendered_rgb_image = self.environment.physics.render(height=160, width=240)
                self.images.append(rendered_rgb_image)
        Logger.log(f"total episode reward: {self.total_reward}", episode=Run.instance().dynamic_config.current_episode, 
                   log_type=Logger.REWARD_TYPE , print_message=True)
        if visualize:
            self.visualize()
        return torch.cat(self.memory , dim=0)
    
    def visualize(self):
        run = Run.instance()
        path = f"{run.experiment_path}/visualizations/{run.dynamic_config.current_episode}"
        makedirs(path, exist_ok=True)
        media.write_video(f"{path}/video.mp4",self.images, fps=30)
    
    @torch.no_grad
    def calculate_advantages(self , memory:TensorDict):
        rewards = memory['reward']
        # rewards = rewards - rewards.mean()
        # rewards = rewards/rewards.std()
        done = torch.zeros_like(rewards)
        done[-1] = 1
        done = done.to(torch.bool)
        current_state_values = memory['current_state_value']
        next_state_values = memory['next_state_value']
        advantage, value_target = generalized_advantage_estimate(Run.instance().ppo_config.gamma, Run.instance().ppo_config.lmbda,
                                                                 current_state_values, next_state_values , rewards, done , done)
        
        memory['current_state_value_target'] = value_target
        memory['advantage'] = advantage
            
        



