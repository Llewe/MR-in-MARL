from enum import Enum


class EnvType(str, Enum):
    SIMPLE = "simple"
    SIMPLE_TAG = "simple_tag"
    SIMPLE_SPREAD = "simple_spread"
    SIMPLE_ADVERSARY = "simple_adversary"

    COIN_GAME = "coin_game"

    P_SIMPLE = "p_simple"
    P_SIMPLE_TAG = "p_simple_tag"
    P_SIMPLE_SPREAD = "p_simple_spread"
    P_SIMPLE_ADVERSARY = "p_simple_adversary"
