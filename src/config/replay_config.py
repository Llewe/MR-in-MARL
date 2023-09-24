from pydantic_settings import BaseSettings

from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(BaseSettings):
    AGENT_TYPE: AgentType = AgentType.ACTOR_CRITIC
    ENV_NAME: EnvType = EnvType.COIN_GAME

    EXPERIMENT_NAME: str = "2023-09-17T15:37:42 - Chris"
    TIMEOUT: int = 100
    EPISODE: int = 10001
    STEPS: int = 500


replay_config = ReplayConfig()
