from pydantic_settings import BaseSettings

from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(BaseSettings):
    AGENT_TYPE: AgentType = AgentType.RANDOM
    ENV_NAME: EnvType = EnvType.SIMPLE
    ENV_TAG: str = "4p 5x5 150c"
    EXPERIMENT_NAME: str = "2023-10-13T14:37:32 - Dawn"
    TIMEOUT: int = 100
    EPISODE: int = 150000
    STEPS: int = 500


replay_config = ReplayConfig()
