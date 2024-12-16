from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from simulation_config import SimulationConfig


class LevyTypeProcess(ABC):
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # Precompute interpolation mappings for tau values
        grid = self._create_precomputed_time_line()
        self.precomputed_tau = {
            1: self._build_tau_interpolator(grid, 1),
            -1: self._build_tau_interpolator(grid, -1),
        }

    def _create_precomputed_time_line(self) -> np.ndarray:
        """Create a time grid with a uniform section on [0, T] and logarithmic on [T, T + 1000]."""
        total_time = self.config.total_time
        n_points = self.config.num_precompute_points
        dense_range = np.linspace(0, total_time, n_points)
        sparse_range = np.logspace(np.log10(total_time), np.log10(total_time + 1000), n_points)
        return np.unique(np.concatenate([dense_range, sparse_range]))

    def _build_tau_interpolator(self, time_grid: np.ndarray, sign: int):
        """Construct interpolation mapping s -> tau((s * h)^eps)."""
        h, eps = self.config.h, self.config.eps
        tau_values = [self.tau((t * h) ** eps, sign) for t in time_grid]
        return interp1d(time_grid, tau_values, kind="quadratic", assume_sorted=True)

    @abstractmethod
    def drift_coefficient(self, t: float | np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
        """Compute the drift coefficient at time t and state x."""
        pass

    @abstractmethod
    def diffusion_coefficient(self, t: float | np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
        """Compute the diffusion coefficient at time t and state x."""
        pass

    @abstractmethod
    def jump_coefficient(
        self, t: float | np.ndarray, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        """Compute the jump coefficient at time t, state x, and jump size z."""
        pass

    @abstractmethod
    def levy_density(self, z: float | np.ndarray) -> float | np.ndarray:
        """Compute the Lévy density for a given jump size `z`."""
        pass

    @abstractmethod
    def levy_tail_integral(self, r: float, sign: int) -> float:
        """Compute the tail integral of the Lévy measure."""
        pass

    def tau(self, t: float, sign: int) -> float:
        """Solve for `r` such that `levy_tail_integral(r, sign) = 1 / t`."""
        if t <= 0:
            return 0.0

        def objective(r: float) -> float:
            return self.levy_tail_integral(r, sign) - 1 / t

        sol = root_scalar(objective, method="brentq", bracket=[0, self.config.brentq_upper_bound])
        return sol.root

    def tau_precomputed(self, t: float | np.ndarray, sign: int) -> float | np.ndarray:
        """Retrieve interpolated tau values from precomputed data."""
        return self.precomputed_tau[sign](t)

    def inverse_large_jump_cdf(self, u: float, t: float, sign: int) -> float:
        """Compute the inverse CDF of the large-jump distribution."""
        h, eps = self.config.h, self.config.eps
        if u >= eps:
            val = ((t * h) ** eps) * ((u / eps) ** (eps / (1 - eps)))
        else:
            val = (1 - eps) * ((t * h) ** eps) / (1 - u)
        try:
            return self.tau_precomputed((val ** (1 / eps)) / h, sign)  # val = (th)^eps  =>  t = val^(1/eps) / h
        except ValueError:
            return self.tau(val, sign)

    def sample_large_jump(self, t: float, sign: int) -> float:
        """Sample a large jump for a given time t and sign."""
        u = self.rng.uniform()
        return sign * self.inverse_large_jump_cdf(u, t, sign)

    def inverse_lambda(self, x: float) -> float:
        """Compute the inverse of the large-jump intensity function."""
        h, eps = self.config.h, self.config.eps
        return (x * (1 - eps) * (h**eps)) ** (1 / (1 - eps))

    def generate_large_jump_times(self, t: float) -> np.ndarray:
        """Generate times for large jumps within the interval [0, t]."""
        jump_times = []
        exp_sum = self.rng.exponential()
        current_time = self.inverse_lambda(exp_sum)
        while 0 < current_time < t:
            jump_times.append(current_time)
            exp_sum += self.rng.exponential()
            current_time = self.inverse_lambda(exp_sum)
        return np.array(jump_times)

    def compute_small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        """Approximate variance of small jumps in [t_prev, t_curr] using trapezoidal rule."""
        ds_grid = np.linspace(t_prev, t_curr, self.config.num_ds_points)
        xi_grid = np.linspace(self.config.min_value, 1.0, self.config.num_dz_points)

        tau_pos = self.tau_precomputed(ds_grid, 1)
        tau_neg = self.tau_precomputed(ds_grid, -1)

        s_mesh, xi_mesh = np.meshgrid(ds_grid, xi_grid, indexing="ij")
        z_pos = xi_mesh * tau_pos[:, None]
        z_neg = -xi_mesh * tau_neg[:, None]

        jump_sq_pos = self.jump_coefficient(s_mesh, x_prev, z_pos) ** 2
        jump_sq_neg = self.jump_coefficient(s_mesh, x_prev, z_neg) ** 2

        levy_pos = self.levy_density(z_pos)
        levy_neg = self.levy_density(z_neg)

        integrand_pos = jump_sq_pos * levy_pos * tau_pos[:, None]
        integrand_neg = jump_sq_neg * levy_neg * tau_neg[:, None]

        integrand_total = integrand_pos + integrand_neg

        return np.trapz(np.trapz(integrand_total, xi_grid, axis=1), ds_grid)
