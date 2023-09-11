from pydantic_settings import BaseSettings


class ActorCriticConfig(BaseSettings):
    TORCH_DEVICE: str = "cpu"

    ACTOR_LR: float = 0.0002
    CRITIC_LR: float = 0.0001
    ACTOR_HIDDEN_UNITS: int = 64
    CRITIC_HIDDEN_UNITS: int = 64
    DISCOUNT_FACTOR: float = 0.95

    CLIP_NORM: float = 1.0


actor_critic_config = ActorCriticConfig()
