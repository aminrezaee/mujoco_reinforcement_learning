from abc import ABC

from tensordict import TensorDict


class BaseAgent(ABC):

    def __init__(self):
        pass

    def act(self, state, return_dist: bool = False):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def get_state_value(self, state) -> float:
        pass
