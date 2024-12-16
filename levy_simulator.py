import warnings

import numpy as np

from levy_type_process import LevyTypeProcess

warnings.filterwarnings("ignore")


class LevySimulator:
    def __init__(self, process: LevyTypeProcess, approximate_small_jumps: bool = True):
        self.process = process
        self.approximate_small_jumps = approximate_small_jumps

    def _build_time_grid_and_jumps(self) -> tuple[np.ndarray, dict[float, float]]:
        """Construct the combined time grid and a mapping of jump times to jump sizes."""
        cfg = self.process.config

        jump_times = []
        jump_dict = {}
        for sign in (1, -1):
            sign_times = self.process.generate_large_jump_times(cfg.total_time)
            jump_times.extend(sign_times)
            jump_dict.update({t: self.process.sample_large_jump(t, sign) for t in sign_times})

        regular_times = np.linspace(cfg.min_value, cfg.total_time, cfg.num_steps)
        full_time = np.concatenate([regular_times, jump_times])
        return np.unique(np.sort(full_time)), jump_dict

    def simulate_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a single path of the Levy-type process."""
        rng = self.process.rng

        times, jump_dict = self._build_time_grid_and_jumps()
        values = np.zeros_like(times)
        values[0] = self.process.config.x_0

        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            t_prev, t_curr = times[i - 1].item(), times[i].item()
            x_prev = values[i - 1].item()

            drift_inc = self.process.drift_coefficient(t_prev, x_prev) * dt
            diffusion_inc = self.process.diffusion_coefficient(t_prev, x_prev) * np.sqrt(dt) * rng.standard_normal()

            large_jump_inc = jump_dict.get(t_curr, 0.0)
            if large_jump_inc:
                large_jump_inc = self.process.jump_coefficient(t_prev, x_prev, large_jump_inc)

            small_jump_inc = 0.0
            if self.approximate_small_jumps:
                small_jump_var = self.process.compute_small_jump_variance(x_prev, t_prev, t_curr)
                small_jump_inc = np.sqrt(small_jump_var) * rng.standard_normal()

            values[i] = x_prev + drift_inc + diffusion_inc + large_jump_inc + small_jump_inc

        return times, values

    def simulate_paths(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Simulate multiple paths of the Levy-type process."""
        return [self.simulate_path() for _ in range(n)]
