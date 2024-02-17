from typing import List

from src.config.ctrl_config import ACConfig
from src.enums.metrics_e import MetricsE


class MaAcConfig(ACConfig):
    UPPER_BOUND: float = 2.0
    LOWER_BOUND: float = 0.0

    NR_MA_AGENTS: int = 1

    PERCENTAGE_MODE: bool = True

    UPDATE_EVERY_X_EPISODES: int = 5


class IndividualMaACPGlobalMetricConfig(MaAcConfig):
    METRICS: List[MetricsE] = [MetricsE.EFFICIENCY]


class CentralMaFixedPercentageConfig(ACConfig):
    PERCENTAGE: float = 0.8
