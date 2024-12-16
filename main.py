import matplotlib.pyplot as plt
import numpy as np

from levy_simulator import LevySimulator
from simulation_config import SimulationConfig
from tempered_stable_levy_type_process import TemperedStableLevyTypeProcess


def plot_process_path(times: np.ndarray, values: np.ndarray) -> None:
    plt.figure(figsize=(12, 6))
    plt.step(times, values, where="post", label="Simulated Path")
    plt.xlabel("Time")
    plt.ylabel("X(t)")
    plt.title("Levy-Type Process - Single Path")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_process_paths(paths: list[tuple[np.ndarray, np.ndarray]]) -> None:
    plt.figure(figsize=(12, 6))
    for times, values in paths:
        plt.step(times, values, where="post", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Levy-Type Process - Multiple Paths")
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

    process = TemperedStableLevyTypeProcess(config=config, alpha=0.75, c=1.0)

    simulator = LevySimulator(process=process, approximate_small_jumps=True)

    n_paths = 100
    paths = simulator.simulate_paths(n=n_paths)

    plot_process_paths(paths)


if __name__ == "__main__":
    main()
