from enum import Enum


class ManipulationMode(str, Enum):
    NONE: str = "none"
    CENTRAL_HEURISTIC: str = "central_heuristic"
    CENTRAL_FIXED_PERCENTAGE: str = "central_fixed_percentage"
    CENTRAL_AC_PERCENTAGE: str = "central_ac_percentage"
    INDIVIDUAL_AC_P_GLOBAL_METRIC: str = "individual_ac_p_global_metric"
    INDIVIDUAL_AC_PERCENTAGE: str = "individual_ac_percentage"
