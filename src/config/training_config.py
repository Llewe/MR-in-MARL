from datetime import datetime
from time import time

import names

from src.cfg_manager import ExpConfig
from src.enums import AgentType, EnvType


class TrainingConfig(ExpConfig):
    EXP_TAG: str = names.get_first_name()

    EPISODES: int = 10
    EPOCHS: int = 50000

    EVAL_EPISODES: int = 250
    EVAL_EPOCH_INTERVAL: int = 500

    AGENT_TYPE: AgentType = AgentType.MATE

    ENV_NAME: EnvType = EnvType.MY_COIN_GAME
    ENV_TAG: str = "default"

    EXP_UNIQUE_NAME: str = (
        f"{datetime.fromtimestamp(time()).isoformat(timespec='seconds')} - {EXP_TAG}"
    )
