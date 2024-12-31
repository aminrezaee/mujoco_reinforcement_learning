from entities.agents.agent import Agent
from dm_control import suite
import numpy as np
import torch
from tensordict import TensorDict
from utils.logger import Logger

class EnvironmentHelper:
    def __init__(self):
        self.environment = suite.load(domain_name="humanoid", task_name="run", task_kwargs={"random": 42})
        self.spec = self.environment.action_spec()
        self.timestep = self.environment.reset()
        self.total_reward = 0
        self.memory = []
        
    def reset(self):
        self.timestep = self.environment.reset()
        self.total_reward = 0
        self.memory = []
        
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
        return next_data
    
    def rollout(self , agent:Agent):
        next_state = self.get_state()
        while not self.timestep.last():
            current_state = torch.clone(next_state)
            current_state_value = agent.get_state_value(current_state)
            action_log_prob , action = agent.act(current_state , return_log_prob=True)
            self.timestep = self.step(action)
            next_state = self.get_state()
            next_state_value = agent.get_state_value(next_state)
            memory_item = {'current_state':current_state[None,:],
                           'current_state_value':current_state_value[None,:] , 
                           'next_state_value':next_state_value[None,:], 
                           'action':action[None,:], 
                           'action_log_prob':action_log_prob[None,:],
                           'reward':self.timestep.reward[None,:]}
            self.memory.append(TensorDict(memory_item,batch_size=1))
        Logger.log("total episode reward: ", self.total_reward)
        return torch.cat(self.memory , dim=0)
    
    def calculate_advantages(self , memory:TensorDict):
        rewards = memory['reward']
        current_state_values = memory['current_state_value']
        next_state_values = memory['next_state_value']
        
        pass
            
if __name__ == "__main__":
    environment_helper = EnvironmentHelper()
    environment_helper.main()
        



