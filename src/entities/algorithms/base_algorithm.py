from environments.helper import EnvironmentHelper as Helper
from entities.agents.agent import Agent


class Algorithm:

    def __init__(self, environment_helper: Helper, agent: Agent):
        self.environment_helper = environment_helper
        self.agent = agent
        pass
