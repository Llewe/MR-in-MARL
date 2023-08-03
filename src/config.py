from pydantic_settings import BaseSettings

from agents.agent_type_e import AgentType


class Config(BaseSettings):
    LOG_LEVEL: str = "INFO"


class EvalConfig(BaseSettings):
    NUM_EPISODES: int = 500
    EVAL_INTERVAL: int = 500


class ActorCriticConfig(BaseSettings):
    TORCH_DEVICE: str = "cpu"

    ACTOR_LR: float = 0.0005
    CRITIC_LR: float = 0.0001
    ACTOR_HIDDEN_UNITS: int = 64
    CRITIC_HIDDEN_UNITS: int = 128
    DISCOUNT_FACTOR: float = 0.95

    CLIP_NORM: float = 1.0


class TrainingConfig(BaseSettings):
    EPISODES: int = 10000
    EPOCHS: int = 1
    AGENT_TYPE: AgentType = AgentType.ACTOR_CRITIC


class EnvConfig(BaseSettings):
    ENV_NAME: str = "simple"
    PARALLEL_ENV: bool = False
    RENDER_MODE: str = ""
    RENDER_FPS: int = 30

    MAX_CYCLES: int = 25
    CONTINUOUS_ACTIONS: bool = False


config = Config()
actor_critic_config = ActorCriticConfig()
training_config = TrainingConfig()
env_config = EnvConfig()
eval_config = EvalConfig()
