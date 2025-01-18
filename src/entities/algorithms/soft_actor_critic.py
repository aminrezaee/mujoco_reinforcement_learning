import torch
from .base_algorithm import Algorithm
from entities.features import Run
from entities.timestep import Timestep
from tensordict import TensorDict
import numpy as np
from torch.nn.functional import huber_loss, mse_loss


class SoftActorCritic(Algorithm):

    def train(self, memory: TensorDict):
        pass

    def _iterate(self):
        pass
