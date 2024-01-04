from enum import Enum


class AgentType(str, Enum):
    RANDOM = "random"
    A2C = "a2c"
    MATE = "mate"
    GIFTING = "gifting"
    LOLA_PG = "lola_pg"

    DEMO_MANIPULATION_AGENT = "demo_manipulation_agent"
    DEMO_MANIPULATION_COIN = "demo_manipulation_coin"

    MA_IPD_PUNISH_DEFECT = "ma_ipd_punish_defect"

    MA_COIN_TO_MIDDLE = "ma_coin_to_middle"
