from enum import Enum


class EnvType(str, Enum):
    SIMPLE = "simple"
    SIMPLE_TAG = "simple_tag"
    SIMPLE_SPREAD = "simple_spread"
    SIMPLE_ADVERSARY = "simple_adversary"

    COIN_GAME = "coin_game"
    MY_COIN_GAME = "my_coin_game"
    P_MY_COIN_GAME = "p_my_coin_game"

    P_HARVEST = "p_harvest"

    PRISONERS_DILEMMA = "prisoners_dilemma"
    SAMARITANS_DILEMMA = "samaritans_dilemma"
    STAG_HUNT = "stag_hunt"
    CHICKEN = "chicken"

    P_SIMPLE = "p_simple"
    P_SIMPLE_TAG = "p_simple_tag"
    P_SIMPLE_SPREAD = "p_simple_spread"
    P_SIMPLE_ADVERSARY = "p_simple_adversary"
