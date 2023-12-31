from src.cfg_manager import ExpConfig
from src.enums.agent_type_e import AgentType
from src.enums.env_type_e import EnvType


class ReplayConfig(ExpConfig):
    LOG_LEVEL: str = "INFO"

    AGENT_TYPE: AgentType = AgentType.MA_COIN_TO_MIDDLE
    ENV_NAME: EnvType = EnvType.P_MY_COIN_GAME
    ENV_TAG: str = "default"
    EXP_UNIQUE_NAME: str = "2023-12-31T08:44:02 - Matthew"
    TIMEOUT: int = 0  # 100
    EPOCH: int = 16500
    STEPS: int = 50000
