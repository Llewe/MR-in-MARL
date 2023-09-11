import names

from pydantic_settings import BaseSettings

from agents.agent_type_e import AgentType
from src.enviroments.enums.env_type_e import EnvType


class ReplayConfig(BaseSettings):
    AGENT_TYPE: AgentType = AgentType.DEMO_MANIPULATION_AGENT
    ENV_NAME: EnvType = EnvType.SIMPLE

    EXPERIMENT_NAME: str = "2023-08-09T15:00:49 - Nicholas"
    TIMEOUT: int = 6
    EPISODE: int = 1000
    STEPS: int = 25


class Config(BaseSettings):
    LOG_LEVEL: str = "INFO"

    NAME_TAG: str = names.get_first_name()


class EvalConfig(BaseSettings):
    NUM_EPISODES: int = 500
    EVAL_INTERVAL: int = 500


class ActorCriticConfig(BaseSettings):
    TORCH_DEVICE: str = "cpu"

    ACTOR_LR: float = 0.0002
    CRITIC_LR: float = 0.0001
    ACTOR_HIDDEN_UNITS: int = 64
    CRITIC_HIDDEN_UNITS: int = 64
    DISCOUNT_FACTOR: float = 0.95

    CLIP_NORM: float = 1.0


class TrainingConfig(BaseSettings):
    EPISODES: int = 100
    EPOCHS: int = 1
    AGENT_TYPE: AgentType = AgentType.DEMO_MANIPULATION_AGENT


class EnvConfig(BaseSettings):
    ENV_NAME: EnvType = EnvType.SIMPLE
    RENDER_MODE: str = ""
    RENDER_FPS: int = 30

    MAX_CYCLES: int = 25
    CONTINUOUS_ACTIONS: bool = False


config = Config()
actor_critic_config = ActorCriticConfig()
training_config = TrainingConfig()
env_config = EnvConfig()
eval_config = EvalConfig()

replay_config = ReplayConfig()
