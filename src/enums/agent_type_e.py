from enum import Enum


class AgentType(str, Enum):
    RANDOM = "random"
    ACTOR_CRITIC = "actor_critic"
    MATE = "mate"
    GIFTING = "gifting"
    LOLA_PG = "lola_pg"

    DEMO_MANIPULATION_AGENT = "demo_manipulation_agent"

    MA_IPD_PUNISH_DEFECT = "ma_ipd_punish_defect"
    MA_IPD_EXPERIMENTS = "ma_ipd_experiments"

    MA_COIN_TO_MIDDLE = "ma_coin_to_middle"
