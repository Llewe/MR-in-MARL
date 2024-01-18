from datetime import datetime
from enum import Enum
from time import time

import names

from src.cfg_manager import ExpConfig
from src.enums import AgentType, EnvType
from src.enums.manipulation_modes_e import ManipulationMode


class TrainingConfig(ExpConfig):
    # EXP_TAG: str = names.get_first_name()
    EXP_TAG: str = "CENTRAL_AC_PERCENTAGE [-1-1] - double-update"
    EPISODES: int = 10
    EPOCHS: int = 1000

    EVAL_EPISODES: int = 100
    EVAL_EPOCH_INTERVAL: int = 50

    AGENT_TYPE: AgentType = AgentType.ACTOR_CRITIC
    ENV_NAME: EnvType = EnvType.P_COIN_GAME
    ENV_TAG: str = "test_4"

    MA_MODE: ManipulationMode = ManipulationMode.CENTRAL_AC_PERCENTAGE

    EXP_UNIQUE_NAME: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self.EXP_UNIQUE_NAME = f"{datetime.fromtimestamp(time()).isoformat(timespec='seconds')} - {self.EXP_TAG}"
