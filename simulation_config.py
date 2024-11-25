from dataclasses import dataclass


@dataclass
class SimulationConfig:
    total_time: float
    num_steps: int
    h: float
    eps: float
    x_0: float = 0.0
    random_seed: int = 42
