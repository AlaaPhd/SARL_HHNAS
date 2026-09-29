from dataclasses import dataclass

@dataclass
class Config:
    dataset: str = 'cifar10'
    budget: int = 100
    seed: int = 42
    beta: float = 0.0
    discovery_bonus: float = 0.05
    utility_alpha: float = 0.3
    history_window: int = 20
    stagnation_max: int = 20
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 10000
    hidden: int = 128
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    target_update: int = 100
    grad_clip: float = 1.0
    warmup: int = 64
    t0: float = 0.20
    tmin: float = 0.01
    params_scale: float = 1.0
    flops_scale: float = 1.0
    state_mask: tuple = ()
    shuffle_state: bool = False
