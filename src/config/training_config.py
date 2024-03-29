from datetime import datetime
from time import time

from src.cfg_manager import ExpConfig
from src.controller_ma.utils.ma_ac import MaAcConfig
from src.enums import AgentType, EnvType
from src.enums.manipulation_modes_e import ManipulationMode


class TrainingConfig(ExpConfig):
    # EXP_TAG: str = names.get_first_name()
    EXP_TAG: str = ""
    EPISODES: int = 10
    EPOCHS: int = 5000

    EVAL_EPISODES: int = 100
    EVAL_EPOCH_INTERVAL: int = 50

    AGENT_TYPE: AgentType = AgentType.ACTOR_CRITIC
    ENV_NAME: EnvType = EnvType.P_COIN_GAME
    ENV_TAG: str = "test_4-n"

    MA_MODE: ManipulationMode = ManipulationMode.CENTRAL_AC_PERCENTAGE

    EXP_UNIQUE_NAME: str = ""

    REST_SEED: int = 42

    def __init__(self, **data):
        super().__init__(**data)

        if self.MA_MODE != ManipulationMode.NONE:
            if self.EXP_TAG == "":
                self.EXP_TAG = f"{self.MA_MODE.value}"
            else:
                self.EXP_TAG = f"{self.EXP_TAG} - {self.MA_MODE.value}"

        if (
            self.MA_MODE == ManipulationMode.CENTRAL_AC_PERCENTAGE
            or self.MA_MODE == ManipulationMode.INDIVIDUAL_AC_P_GLOBAL_METRIC
            or self.MA_MODE == ManipulationMode.INDIVIDUAL_AC_PERCENTAGE
        ):
            cfg = MaAcConfig()
            self.EXP_TAG = f"{self.EXP_TAG} - [{cfg.LOWER_BOUND}-{cfg.UPPER_BOUND}] - Disc{cfg.DISCOUNT_FACTOR}"

            if cfg.UPDATE_EVERY_X_EPISODES > 0:
                self.EXP_TAG = f"{self.EXP_TAG} - UPDATE={cfg.UPDATE_EVERY_X_EPISODES}"
        self.EXP_TAG = f"{self.EXP_TAG} - Seed={self.REST_SEED}"
        self.EXP_UNIQUE_NAME = f"{datetime.fromtimestamp(time()).isoformat(timespec='seconds')} - {self.EXP_TAG}"
