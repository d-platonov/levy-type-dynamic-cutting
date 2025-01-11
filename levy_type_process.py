from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from simulation_config import SimulationConfig


class LevyTypeProcess(ABC):
    """Abstract base class for simulating a Lévy-type process using dynamic cutting."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # Precompute interpolation mappings for tau((s*h)^eps) for sign +1 and -1
        grid = self._create_precomputed_time_line()
        self.precomputed_tau = {
            +1: self._build_tau_interpolator(grid, +1),
            -1: self._build_tau_interpolator(grid, -1),
        }

    def _create_precomputed_time_line(self) -> np.ndarray:
        """
        Create a time grid for interpolation of tau.
            - A uniform grid on [0, total_time].
            - A log-spaced extension from [total_time, total_time+1000].
        """
        total_time = self.config.total_time
        n_points = self.config.num_precompute_points

        dense_range = np.linspace(0, total_time, n_points)
        sparse_range = np.logspace(
            np.log10(total_time),
            np.log10(total_time + 1000),
            n_points,
        )

        grid = np.unique(np.concatenate([dense_range, sparse_range]))
        return grid

    def _build_tau_interpolator(self, time_grid: np.ndarray, sign: int):
        """Construct interpolation for s -> tau((sh)^eps)."""
        h, eps = self.config.h, self.config.eps
        scaled_times = (time_grid * h) ** eps
        tau_values = [self.tau(st, sign) for st in scaled_times]
        return interp1d(
            time_grid,
            tau_values,
            kind="quadratic",
            assume_sorted=True,
        )

    @abstractmethod
    def drift_coefficient(self, t: float | np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
        """Compute the drift coefficient a(t, x)."""
        pass

    @abstractmethod
    def diffusion_coefficient(self, t: float | np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
        """Compute the diffusion coefficient b(t, x)."""
        pass

    @abstractmethod
    def jump_coefficient(
        self, t: float | np.ndarray, x: float | np.ndarray, z: float | np.ndarray
    ) -> float | np.ndarray:
        """Compute the jump coefficient c(t, x, z)."""
        pass

    @abstractmethod
    def levy_density(self, z: float | np.ndarray) -> float | np.ndarray:
        """Lévy density. Must be integrable on R without {0}."""
        pass

    @abstractmethod
    def levy_tail_integral(self, r: float, sign: int) -> float:
        """
        Compute one of the tail integrals:
            N^+(r) = int_{+r}^{+inf} levy_density(z) dz (sign=+1),
            N^-(r) = int_{-inf}^{-r} levy_density(z) dz (sign=-1).
        """
        pass

    @lru_cache(None)
    def tau(self, t: float, sign: int) -> float:
        """Solve r such that levy_tail_integral(r, sign) = 1 / t, via Brent's method."""
        if t <= 0:
            return 0.0

        def objective(r: float) -> float:
            return self.levy_tail_integral(r, sign) - 1.0 / t

        sol = root_scalar(
            objective,
            method="brentq",
            bracket=[0, self.config.brentq_upper_bound],
        )
        return sol.root

    def tau_precomputed(self, s: float | np.ndarray, sign: int) -> float | np.ndarray:
        """Interpolate precomputed tau((s*h)^eps)."""
        return self.precomputed_tau[sign](s)

    def inverse_large_jump_cdf(self, u: float, t: float, sign: int) -> float:
        """Inverse CDF for the large-jump distribution."""
        h, eps = self.config.h, self.config.eps

        if u >= eps:
            val = ((t * h) ** eps) * ((u / eps) ** (eps / (1 - eps)))
        else:
            val = (1 - eps) * ((t * h) ** eps) / (1 - u)

        try:
            return self.tau_precomputed((val ** (1.0 / eps)) / h, sign)
        except ValueError:
            return self.tau(val, sign)  # If out-of-range, do the direct rootfinding

    def sample_large_jump(self, t: float, sign: int) -> float:
        """Sample one large jump (size) at time t and sign (+/-)."""
        u = self.rng.uniform()
        jump_size = self.inverse_large_jump_cdf(u, t, sign)
        return sign * jump_size

    def inverse_lambda(self, x: float) -> float:
        """Inverse of the large-jump intensity function λ^{±}(t)."""
        h, eps = self.config.h, self.config.eps
        return (x * (1 - eps) * (h**eps)) ** (1.0 / (1 - eps))

    def generate_large_jump_times(self, t: float) -> np.ndarray:
        """Generate jump times for large jumps in [0, t]."""
        jump_times = []
        exp_sum = self.rng.exponential()
        current_time = self.inverse_lambda(exp_sum)
        while 0 < current_time < t:
            jump_times.append(current_time)
            exp_sum += self.rng.exponential()
            current_time = self.inverse_lambda(exp_sum)
        return np.array(jump_times)

    def compute_small_jump_variance(self, x_prev: float, t_prev: float, t_curr: float) -> float:
        """Approximate the variance of small jumps in [t_prev, t_curr] by a 2D trapezoidal rule."""
        ds_grid = np.linspace(t_prev, t_curr, self.config.num_ds_points)
        xi_grid = np.linspace(self.config.min_value, 1.0, self.config.num_dz_points)

        tau_pos = self.tau_precomputed(ds_grid, +1)
        tau_neg = self.tau_precomputed(ds_grid, -1)

        # Create 2D meshes
        s_mesh, xi_mesh = np.meshgrid(ds_grid, xi_grid, indexing="ij")

        # Map xi in [0,1] to actual jump sizes
        # For positive side: z in [0, tau_pos]; for negative: z in [-tau_neg, 0]
        z_pos = +xi_mesh * tau_pos[:, None]  # shape matches s_mesh
        z_neg = -xi_mesh * tau_neg[:, None]

        # c(t, x_prev, z)^2
        jump_sq_pos = self.jump_coefficient(s_mesh, x_prev, z_pos) ** 2
        jump_sq_neg = self.jump_coefficient(s_mesh, x_prev, z_neg) ** 2

        levy_pos = self.levy_density(z_pos)
        levy_neg = self.levy_density(z_neg)

        integrand_pos = jump_sq_pos * levy_pos * tau_pos[:, None]
        integrand_neg = jump_sq_neg * levy_neg * tau_neg[:, None]

        integrand_total = integrand_pos + integrand_neg

        val_over_xi = np.trapz(integrand_total, x=xi_grid, axis=1)
        val_2d = np.trapz(val_over_xi, x=ds_grid)

        return val_2d
