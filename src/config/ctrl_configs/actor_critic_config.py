from src.config.ctrl_configs.ctrl_config import CtrlConfig


class ActorCriticConfig(CtrlConfig):

    ACTOR_LR: float = 0.001
    CRITIC_LR: float = 0.001
    ACTOR_HIDDEN_UNITS: int = 64
    CRITIC_HIDDEN_UNITS: int = 64
    DISCOUNT_FACTOR: float = 0.99

    CLIP_NORM: float = 1.0

    REWARD_NORMALIZATION: bool = False

    EPSILON_INIT: float = 0.6
    EPSILON_MIN: float = 0.1
    EPSILON_DECAY: float = 0.0005




