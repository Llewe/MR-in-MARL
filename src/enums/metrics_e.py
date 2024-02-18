from enum import Enum


class MetricsE(str, Enum):
    """
    Enum for the metrics.
    """

    EFFICIENCY: str = "efficiency"
    PEACE: str = "peace"
