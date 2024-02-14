from enum import Enum


class AgentType(str, Enum):
    RANDOM = "random"
    ACTOR_CRITIC = "actor_critic"
    MATE = "mate"
    GIFTING = "gifting"
    LOLA_PG = "lola_pg"

    MA_IPD_PUNISH_DEFECT = "ma_ipd_punish_defect"

    MA_COIN_TO_MIDDLE_AC = "ma_coin_to_middle_ac"
    MA_COIN_TO_MIDDLE_MATE = "ma_coin_to_middle_mate"
    MA_SOCIAL_WELLFARE_AC = "ma_social_wellfare_ac"
    MA_SOCIAL_WELLFARE_MATE = "ma_social_wellfare_mate"
    MA_COIN_DONT_TAKE_OTHERS_AC = "ma_dont_take_others_ac"
    MA_COIN_DONT_TAKE_OTHERS_MATE = "ma_dont_take_others_mate"

    MA_HARVEST_OPTIMAL_PATH_AC = "ma_harvest_optimal_path_ac"
    MA_HARVEST_OPTIMAL_PATH_MATE = "ma_harvest_optimal_path_mate"
