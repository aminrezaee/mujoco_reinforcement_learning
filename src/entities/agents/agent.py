from abc import ABC

from tensordict import TensorDict
class Agent(ABC):
    
    def __init__(self):
        pass
    
    def act(self, state , return_log_probs:bool=False):
        pass
    
    def train(self , memory:TensorDict):
        pass
    
    def save(self):
        pass
    
    def load(self):
        pass
    
    def get_state_value(self, state) -> float:
        pass
    
    
