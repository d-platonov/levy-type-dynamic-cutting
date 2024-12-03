from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from simulation_config import SimulationConfig


class LevyTypeProcess(ABC):
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    @abstractmethod
    def drift_coefficient(self, t: float, x: float) -> float:
        pass

    @abstractmethod
    def diffusion_coefficient(self, t: float, x: float) -> float:
        pass

    @abstractmethod
    def jump_coefficient(self, t: float, x: float, z: float) -> float:
        pass

    @abstractmethod
    def levy_density(self, z: float) -> float:
        pass

    @abstractmethod
    def levy_tail_integral(self, r: float, sign: int) -> float:
        """Compute N^+(r) or N^-(r) for the given sign."""
        pass

    def tau(self, t: float, sign: int) -> float:
        """Compute tau such that N(tau) = 1 / t."""

        # TODO: move hardcoded values
        r_min = 1e-6
        r_max = 100.0
        r_max_threshold = 1e6

        if t <= 0:
            return r_min

        def objective(r: float) -> float:
            return self.levy_tail_integral(r, sign) - (1 / t)

        f_r_min = objective(r_min)
        f_r_max = objective(r_max)
        while f_r_min * f_r_max > 0:
            r_max *= 2
            if r_max > r_max_threshold:
                return r_max
            f_r_max = objective(r_max)

        return brentq(objective, a=r_min, b=r_max, xtol=1e-6)

    def inverse_large_jump_cdf(self, u: float, t: float, sign: int) -> float:
        h, eps = self.config.h, self.config.eps
        if u >= eps:
            return self.tau((t * h) ** eps * (u / eps) ** (eps / (1 - eps)), sign)
        return self.tau((1 - eps) * ((t * h) ** eps) / (1 - u), sign)

    def sample_large_jump(self, t: float, sign: int):
        return sign * self.inverse_large_jump_cdf(self.rng.uniform(), t, sign)

    def inverse_lambda(self, t: float) -> float:
        h, eps = self.config.h, self.config.eps
        return (t * (1 - eps) * (h ** eps)) ** (1 / (1 - eps))

    def generate_jump_times(self, t: float) -> np.ndarray:
        """Inverse time transformation method."""
        jump_times = []
        accumulated_exponential = self.rng.exponential()
        current_time = self.inverse_lambda(accumulated_exponential)
        while 0 < current_time < t:
            jump_times.append(current_time)
            accumulated_exponential += self.rng.exponential()
            current_time = self.inverse_lambda(accumulated_exponential)
        return np.array(jump_times)

    def compute_small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        h, eps = self.config.h, self.config.eps

        def integrand(z: float, s: float) -> float:
            return self.jump_coefficient(s, x_prev, z) ** 2 * self.levy_density(z)

        def small_jump_variance(s: float) -> float:
            tau_s_pos = self.tau((s * h) ** eps, sign=1)
            tau_s_neg = self.tau((s * h) ** eps, sign=-1)
            var_pos = quad(lambda z: integrand(z, s), a=0, b=tau_s_pos)[0]
            var_neg = quad(lambda z: integrand(z, s), a=-tau_s_neg, b=0)[0]
            return var_pos + var_neg

        variance = quad(small_jump_variance, t_prev, t_curr)[0]
        return variance
