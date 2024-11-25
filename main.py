import matplotlib.pyplot as plt
import numpy as np

from levy_simulator import LevySimulator
from simulation_config import SimulationConfig
from tempered_stable_levy_type_process import TemperedStableLevyTypeProcess


def plot_process_path(grid: np.ndarray, values: np.ndarray) -> None:
    plt.figure(figsize=(12, 6))
    plt.step(grid, values, where="post", label="Simulated Path")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.title("Symmetric Tempered Stable Levy-Type Process")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    config = SimulationConfig(
        total_time=10.0,
        num_steps=100,
        h=0.05,
        eps=0.01,
        x_0=0.0,
        random_seed=123,
    )

    process = TemperedStableLevyTypeProcess(config=config, alpha=0.75, c=1)

    simulator = LevySimulator(process)

    time_grid, x_values = simulator.simulate_path()

    plot_process_path(time_grid, x_values)


if __name__ == "__main__":
    main()
