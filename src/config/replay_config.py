from src.cfg_manager import ExpConfig
from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(ExpConfig):
    LOG_LEVEL: str = "INFO"

    AGENT_TYPE: AgentType = AgentType.A2C
    ENV_NAME: EnvType = EnvType.SIMPLE_TAG
    ENV_TAG: str = "1vs3 - 2 obstacles - 150steps"
    EXP_UNIQUE_NAME: str = "2023-11-21T10:35:17 - Robin"
    TIMEOUT: int = 100
    EPOCH: int = 35000
    STEPS: int = 500
