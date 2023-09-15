from pydantic_settings import BaseSettings

from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(BaseSettings):
    AGENT_TYPE: AgentType = AgentType.DEMO_MANIPULATION_AGENT
    ENV_NAME: EnvType = EnvType.COIN_GAME

    EXPERIMENT_NAME: str = "2023-08-07T18:18:46 - Tina"
    TIMEOUT: int = 6
    EPISODE: int = 1000
    STEPS: int = 25


replay_config = ReplayConfig()
