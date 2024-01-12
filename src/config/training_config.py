from datetime import datetime
from enum import Enum
from time import time

import names

from src.cfg_manager import ExpConfig
from src.enums import AgentType, EnvType
from src.enums.manipulation_modes_e import ManipulationMode


class TrainingConfig(ExpConfig):
    # EXP_TAG: str = names.get_first_name()
    EXP_TAG: str = "individual-3ma-0.5-0.5"
    EPISODES: int = 10
    EPOCHS: int = 5000

    EVAL_EPISODES: int = 100
    EVAL_EPOCH_INTERVAL: int = 50

    AGENT_TYPE: AgentType = AgentType.ACTOR_CRITIC
    ENV_NAME: EnvType = EnvType.P_COIN_GAME
    ENV_TAG: str = "new_test"

    MANIPULATION_MODE: ManipulationMode = ManipulationMode.INDIVIDUAL_ACTOR_CRITIC

    EXP_UNIQUE_NAME: str = (
        f"{datetime.fromtimestamp(time()).isoformat(timespec='seconds')} - {EXP_TAG}"
    )
