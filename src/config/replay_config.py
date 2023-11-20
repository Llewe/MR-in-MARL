from src.cfg_manager import ExpConfig
from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(ExpConfig):
    LOG_LEVEL: str = "INFO"

    AGENT_TYPE: AgentType = AgentType.A2C
    ENV_NAME: EnvType = EnvType.MY_COIN_GAME
    ENV_TAG: str = "4x4"
    EXP_UNIQUE_NAME: str = "2023-11-19T15:54:29 - Lindsey"
    TIMEOUT: int = 100
    EPOCH: int = 500
    STEPS: int = 500
