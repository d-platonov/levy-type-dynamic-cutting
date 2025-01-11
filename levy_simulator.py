import warnings
import numpy as np

from levy_type_process import LevyTypeProcess

warnings.filterwarnings("ignore")


class LevySimulator:
    """A simulator for Lévy-type processes using dynamic cutting."""

    def __init__(self, process: LevyTypeProcess, approximate_small_jumps: bool = True):
        self.process = process
        self.approximate_small_jumps = approximate_small_jumps

        # Build and store the deterministic (regular) times
        self._regular_times = np.linspace(
            process.config.min_value,
            process.config.total_time,
            process.config.num_steps,
        )

    def _generate_large_jumps_for_path(self) -> tuple[np.ndarray, dict[float, float]]:
        """Generate large-jump times and sizes for a single path, for both sign=+1 and sign=-1."""
        cfg = self.process.config
        jump_times_list = []
        jump_dict = {}

        for sign in (+1, -1):
            sign_times = self.process.generate_large_jump_times(cfg.total_time)
            jump_times_list.append(sign_times)
            for jt in sign_times:
                raw_jump = self.process.sample_large_jump(jt, sign)
                jump_dict[jt] = raw_jump

        all_jump_times = np.concatenate(jump_times_list) if jump_times_list else np.array([])
        return all_jump_times, jump_dict

    def _build_time_grid_for_path(self) -> tuple[np.ndarray, dict[float, float]]:
        """
        Construct the full time grid for a single path by combining:
            - The fixed regular times (shared across all paths).
            - Generated large jump times (unique per path).
        """
        jump_times, jump_dict = self._generate_large_jumps_for_path()
        full_time = np.concatenate([self._regular_times, jump_times])
        times = np.unique(np.sort(full_time))
        return times, jump_dict

    def simulate_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a single path."""
        rng = self.process.rng
        times, jump_dict = self._build_time_grid_for_path()

        values = np.zeros_like(times, dtype=float)
        values[0] = self.process.config.x_0

        for i in range(1, len(times)):
            x_prev, t_prev, t_curr = values[i - 1], times[i - 1], times[i]
            dt = t_curr - t_prev

            drift_inc = self.process.drift_coefficient(t_prev, x_prev) * dt

            diffusion_inc = (
                    self.process.diffusion_coefficient(t_prev, x_prev)
                    * np.sqrt(dt)
                    * rng.standard_normal()
            )

            large_jump = jump_dict.get(t_curr.item(), 0.0)
            if large_jump != 0.0:
                large_jump_inc = self.process.jump_coefficient(t_prev, x_prev, large_jump)
            else:
                large_jump_inc = 0.0

            small_jump_inc = 0.0
            if self.approximate_small_jumps:
                var_small = self.process.compute_small_jump_variance(x_prev.item(), t_prev.item(), t_curr.item())
                if var_small > 0.0:
                    small_jump_inc = np.sqrt(var_small) * rng.standard_normal()

            values[i] = x_prev + drift_inc + diffusion_inc + large_jump_inc + small_jump_inc

        return times, values

    def simulate_paths(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Simulate n independent paths."""
        return [self.simulate_path() for _ in range(n)]
