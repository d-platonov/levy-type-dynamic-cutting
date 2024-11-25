import numpy as np
from mpmath import gammainc

from levy_type_process import LevyTypeProcess
from simulation_config import SimulationConfig


class TemperedStableLevyTypeProcess(LevyTypeProcess):
    def __init__(self, config: SimulationConfig, alpha: float, c: float):
        super().__init__(config)
        self.alpha = alpha
        self.c = c

    def drift_coefficient(self, t: float, x: float) -> float:
        return -0.1 * x

    def diffusion_coefficient(self, t: float, x: float) -> float:
        return 0.2 * np.sqrt(1 + x ** 2)

    def jump_coefficient(self, t: float, x: float, z: float) -> float:
        return z * (1 + 0.1 * np.abs(x))

    def levy_density(self, z: float) -> float:
        return np.exp(-self.c * z) / (z ** (1 + self.alpha))

    def levy_tail_integral(self, r: float, sign: int) -> float:
        if r <= 0:
            return np.inf
        return float((self.c**self.alpha) * gammainc(-self.alpha, self.c * r, np.inf))
