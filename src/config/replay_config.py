from pydantic_settings import BaseSettings

from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(BaseSettings):
    AGENT_TYPE: AgentType = AgentType.A2C2
    ENV_NAME: EnvType = EnvType.MY_COIN_GAME
    ENV_TAG: str = "2x2"
    EXPERIMENT_NAME: str = "2023-11-17T16:32:22 - Mittie"
    TIMEOUT: int = 100
    EPISODE: int = 5000
    STEPS: int = 500


replay_config = ReplayConfig()
