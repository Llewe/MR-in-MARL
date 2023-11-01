from src.config.ctrl_configs.ctrl_config import CtrlConfig


class A2cConfig(CtrlConfig):
    ACTOR_LR: float = 0.0001
    CRITIC_LR: float = 0.0001
    ACTOR_HIDDEN_UNITS: int = 64
    CRITIC_HIDDEN_UNITS: int = 64
    DISCOUNT_FACTOR: float = 0.95
    UPDATE_FREQ: int = 6

    CLIP_NORM: float = 1

    REWARD_NORMALIZATION: bool = False

    EPSILON_INIT: float = 0.6
    EPSILON_MIN: float = 0.05
    EPSILON_DECAY: float = 1.0e-05