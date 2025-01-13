from dataclasses import dataclass
import numpy as np


@dataclass
class Timestep:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict
