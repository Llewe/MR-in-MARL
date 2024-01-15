from datetime import datetime
from enum import Enum
from time import time

import names

from src.cfg_manager import ExpConfig
from src.enums import AgentType, EnvType
from src.enums.manipulation_modes_e import ManipulationMode


class TrainingConfig(ExpConfig):
    # EXP_TAG: str = names.get_first_name()
    EXP_TAG: str = "INDIVIDUAL_AC_P_GLOBAL_METRIC - [0-10]"
    EPISODES: int = 10
    EPOCHS: int = 15000

    EVAL_EPISODES: int = 100
    EVAL_EPOCH_INTERVAL: int = 50

    AGENT_TYPE: AgentType = AgentType.MATE
    ENV_NAME: EnvType = EnvType.P_HARVEST
    ENV_TAG: str = "test"

    MA_MODE: ManipulationMode = ManipulationMode.INDIVIDUAL_AC_P_GLOBAL_METRIC

    EXP_UNIQUE_NAME: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        self.EXP_UNIQUE_NAME = f"{datetime.fromtimestamp(time()).isoformat(timespec='seconds')} - {self.EXP_TAG}"
