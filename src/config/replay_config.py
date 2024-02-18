from src.cfg_manager import ExpConfig
from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(ExpConfig):
    LOG_LEVEL: str = "INFO"

    AGENT_TYPE: AgentType = AgentType.RANDOM
    ENV_NAME: EnvType = EnvType.P_COIN_GAME
    ENV_TAG: str = "6pl-default-5000"
    EXP_UNIQUE_NAME: str = "2024-01-06T17:02:41 - Katherine"
    TIMEOUT: int = 0  # 100
    EPOCH: int = 3000
    STEPS: int = 5000
