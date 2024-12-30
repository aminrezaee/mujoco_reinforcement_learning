from dm_control import suite
import numpy as np

class EnvironmentHelper:
    def __init__(self):
        self.environment = suite.load(domain_name="humanoid", task_name="run", task_kwargs={"random": 42})
        self.spec = self.environment.action_spec()
        self.timestep = self.environment.reset()
        self.total_reward = 0
        
    def reset(self):
        self.timestep = self.environment.reset()
        self.total_reward = 0
        
    def step(self, action:np.ndarray):
        self.timestep = self.environment.step(action)
        self.total_reward += self.timestep.reward
        return self.timestep, self.total_reward
    
    def main(self):
        while not self.timestep.last():
            action = np.random.uniform(self.spec.minimum, self.spec.maximum, self.spec.shape)
            self.timestep, total_reward = self.step(action)
            print(f"Reward: {self.timestep.reward:.2f} Total Reward: {total_reward:.2f}")
            
if __name__ == "__main__":
    environment_helper = EnvironmentHelper()
    environment_helper.main()
        



