from datetime import datetime
from time import time

import names

from src.cfg_manager import ExpConfig
from src.enums import AgentType, EnvType


class TrainingConfig(ExpConfig):
    EXP_TAG: str = names.get_first_name()

    EPISODES: int = 10
    EPOCHS: int = 5000

    EVAL_EPISODES: int = 100
    EVAL_EPOCH_INTERVAL: int = 50

    AGENT_TYPE: AgentType = AgentType.MATE
    ENV_NAME: EnvType = EnvType.P_COIN_GAME
    ENV_TAG: str = "n-4pl-5000"

    EXP_UNIQUE_NAME: str = (
        f"{datetime.fromtimestamp(time()).isoformat(timespec='seconds')} - {EXP_TAG}"
    )
