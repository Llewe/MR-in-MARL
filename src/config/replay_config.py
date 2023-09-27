from pydantic_settings import BaseSettings

from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(BaseSettings):
    AGENT_TYPE: AgentType = AgentType.A2C2
    ENV_NAME: EnvType = EnvType.COIN_GAME
    ENV_TAG: str = "2p 3x3"
    EXPERIMENT_NAME: str = "2023-09-27T17:47:51 - Samuel"
    TIMEOUT: int = 100
    EPISODE: int = 40000
    STEPS: int = 500


replay_config = ReplayConfig()
