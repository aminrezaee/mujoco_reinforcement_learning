from environments.helper import EnvironmentHelper as Helper
from entities.agents.agent import Agent
from abc import ABC
from tensordict import TensorDict


class Algorithm(ABC):

    def __init__(self, environment_helper: Helper, agent: Agent):
        self.environment_helper = environment_helper
        self.agent = agent
        pass

    def test(self, visualize: bool):
        pass

    def train(self, memory: TensorDict):
        pass
