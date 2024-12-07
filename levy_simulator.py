import warnings

import numpy as np

from levy_type_process import LevyTypeProcess

warnings.filterwarnings("ignore")


class LevySimulator:
    def __init__(self, process: LevyTypeProcess):
        self.process = process

    def prepare_timeline(self) -> tuple[np.ndarray, dict[np.ndarray, np.ndarray]]:
        """Prepare time points and jumps for simulation."""
        times = np.linspace(0, self.process.config.total_time, self.process.config.num_steps)
        jumps_dict = {}
        for sign in (1, -1):
            jump_times = self.process.generate_jump_times(self.process.config.total_time)
            jumps_dict.update({t: self.process.sample_large_jump(t, sign) for t in jump_times})
            times = np.append(times, jump_times)
        return np.sort(times), jumps_dict

    def simulate_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a single path of the Levy-type process."""
        times, jumps_dict = self.prepare_timeline()
        values = np.zeros(len(times))
        values[0] = self.process.config.x_0

        for i in range(1, len(times)):
            dt = times[i] - times[i - 1]
            x_prev = values[i - 1]
            t_prev = times[i - 1]
            t_curr = times[i]

            drift = self.process.drift_coefficient(t_prev, x_prev) * dt

            diffusion = (
                self.process.diffusion_coefficient(t_prev, x_prev)
                * np.sqrt(dt)
                * self.process.rng.standard_normal()
            )

            small_jump_var = self.process.compute_small_jump_variance(x_prev, t_prev, t_curr)
            small_jump_increment = (np.sqrt(small_jump_var) * self.process.rng.standard_normal())

            if t_curr in jumps_dict:
                large_jump_increment = self.process.jump_coefficient(t_prev, x_prev, jumps_dict.get(t_curr))
            else:
                large_jump_increment = 0.0

            values[i] = x_prev + drift + diffusion + small_jump_increment + large_jump_increment

        return times, values
