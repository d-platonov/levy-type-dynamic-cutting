import attrs
from attrs import validators


@attrs.define
class SimulationConfig:
    """
    Configuration parameters for simulating a Levy-type process.

    Attributes:
        total_time (float): Total simulation time horizon (T).
        num_steps (int): Number of discrete time steps for output times (N).
        h (float): Hyperparameter for dynamic cutting.
        eps (float): Hyperparameter for dynamic cutting.
        x_0 (float): Initial value of the process at time t=0.
        random_seed (int): Seed for the random number generator.
        num_precompute_points (int): Points for precomputing tau values.
        num_ds_points (int): Discretization points for integration over s (small-jump variance).
        num_dz_points (int): Discretization points for integration over z (small-jump variance).
        brentq_upper_bound (float): Upper bound for brentq bracket parameter.
        min_value (float): A small value to avoid infinities at 0.
    """

    total_time: float = attrs.field(validator=[validators.instance_of(float), validators.gt(0)])
    num_steps: int = attrs.field(validator=[validators.instance_of(int), validators.gt(0)])
    h: float = attrs.field(validator=[validators.instance_of(float), validators.gt(0)])
    eps: float = attrs.field(validator=[validators.instance_of(float), validators.gt(0), validators.lt(1)])
    x_0: float = attrs.field(default=0.0, validator=validators.instance_of(float))
    random_seed: int = attrs.field(default=42, validator=validators.instance_of(int))
    num_precompute_points: int = attrs.field(default=100, validator=[validators.instance_of(int), validators.gt(0)])
    num_ds_points: int = attrs.field(default=25, validator=[validators.instance_of(int), validators.gt(0)])
    num_dz_points: int = attrs.field(default=25, validator=[validators.instance_of(int), validators.gt(0)])
    brentq_upper_bound: float = attrs.field(default=10.0, validator=validators.instance_of(float))
    min_value: float = attrs.field(default=1e-12, validator=validators.instance_of(float))
