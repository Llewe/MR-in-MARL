from enum import Enum


class EnvType(str, Enum):
    SIMPLE = "simple"
    SIMPLE_TAG = "simple_tag"
    SIMPLE_SPREAD = "simple_spread"
    SIMPLE_ADVERSARY = "simple_adversary"

    COIN_GAME = "coin_game"
    MY_COIN_GAME = "my_coin_game"

    PRISONERS_DILEMMA = "prisoners_dilemma"
    SAMARITANS_DILEMMA = "samaritans_dilemma"
    STAG_HUNT = "stag_hunt"
    CHICKEN = "chicken"

    P_SIMPLE = "p_simple"
    P_SIMPLE_TAG = "p_simple_tag"
    P_SIMPLE_SPREAD = "p_simple_spread"
    P_SIMPLE_ADVERSARY = "p_simple_adversary"

    MELTING_POD_PRISONERS_DILEMMA_IN_THE_MATRIX_ARENA = (
        "prisoners_dilemma_in_the_matrix__arena"
    )
    MELTING_POD_STAG_HUNT_IN_THE_MATRIX_ARENA = "stag_hunt_in_the_matrix__arena"
    MELTING_POD_COLLABORATIVE_COOKING_ASYMMETRIC = "collaborative_cooking__asymmetric"
    MELTING_POD_PURE_COORDINATION_IN_THE_MATRIX_ARENA = (
        "pure_coordination_in_the_matrix__arena"
    )
    MELTING_POD_COINS = "coins"
    MELTING_POD_HIDDEN_AGENDA = "hidden_agenda"
    MELTING_POD_PREDATOR_PREY_RANDOM_FOREST = "predator_prey__random_forest"
    MELTING_POD_CHEMISTRY_TWO_METABOLIC_CYCLES_WITH_DISTRACTORS = (
        "chemistry__two_metabolic_cycles_with_distractors"
    )
    MELTING_POD_RATIONALIZABLE_COORDINATION_IN_THE_MATRIX_ARENA = (
        "rationalizable_coordination_in_the_matrix__arena"
    )
    MELTING_POD_DAYCARE = "daycare"
    MELTING_POD_RUNNING_WITH_SCISSORS_IN_THE_MATRIX_REPEATED = (
        "running_with_scissors_in_the_matrix__repeated"
    )
    MELTING_POD_BACH_OR_STRAVINSKY_IN_THE_MATRIX_REPEATED = (
        "bach_or_stravinsky_in_the_matrix__repeated"
    )
    MELTING_POD_PURE_COORDINATION_IN_THE_MATRIX_REPEATED = (
        "pure_coordination_in_the_matrix__repeated"
    )
    MELTING_POD_COLLABORATIVE_COOKING_CRAMPED = "collaborative_cooking__cramped"
    MELTING_POD_COOP_MINING = "coop_mining"
    MELTING_POD_COMMONS_HARVEST_CLOSED = "commons_harvest__closed"
    MELTING_POD_COLLABORATIVE_COOKING_FORCED = "collaborative_cooking__forced"
    MELTING_POD_PRISONERS_DILEMMA_IN_THE_MATRIX_REPEATED = (
        "prisoners_dilemma_in_the_matrix__repeated"
    )
    MELTING_POD_FACTORY_COMMONS_EITHER_OR = "factory_commons__either_or"
    MELTING_POD_CHEMISTRY_TWO_METABOLIC_CYCLES = "chemistry__two_metabolic_cycles"
    MELTING_POD_TERRITORY_OPEN = "territory__open"
    MELTING_POD_GIFT_REFINEMENTS = "gift_refinements"
    MELTING_POD_ALLELOPATHIC_HARVEST_OPEN = "allelopathic_harvest__open"
    MELTING_POD_CHEMISTRY_THREE_METABOLIC_CYCLES = "chemistry__three_metabolic_cycles"
    MELTING_POD_STAG_HUNT_IN_THE_MATRIX_REPEATED = "stag_hunt_in_the_matrix__repeated"
    MELTING_POD_PREDATOR_PREY_ORCHARD = "predator_prey__orchard"
    MELTING_POD_BACH_OR_STRAVINSKY_IN_THE_MATRIX_ARENA = (
        "bach_or_stravinsky_in_the_matrix__arena"
    )
    MELTING_POD_PAINTBALL_KING_OF_THE_HILL = "paintball__king_of_the_hill"
    MELTING_POD_PREDATOR_PREY_ALLEY_HUNT = "predator_prey__alley_hunt"
    MELTING_POD_TERRITORY_INSIDE_OUT = "territory__inside_out"
    MELTING_POD_CHICKEN_IN_THE_MATRIX_REPEATED = "chicken_in_the_matrix__repeated"
    MELTING_POD_RUNNING_WITH_SCISSORS_IN_THE_MATRIX_ONE_SHOT = (
        "running_with_scissors_in_the_matrix__one_shot"
    )
    MELTING_POD_PAINTBALL_CAPTURE_THE_FLAG = "paintball__capture_the_flag"
    MELTING_POD_COLLABORATIVE_COOKING_CIRCUIT = "collaborative_cooking__circuit"
    MELTING_POD_RATIONALIZABLE_COORDINATION_IN_THE_MATRIX_REPEATED = (
        "rationalizable_coordination_in_the_matrix__repeated"
    )
    MELTING_POD_COLLABORATIVE_COOKING_FIGURE_EIGHT = (
        "collaborative_cooking__figure_eight"
    )
    MELTING_POD_CLEAN_UP = "clean_up"
    MELTING_POD_COMMONS_HARVEST_PARTNERSHIP = "commons_harvest__partnership"
    MELTING_POD_COLLABORATIVE_COOKING_CROWDED = "collaborative_cooking__crowded"
    MELTING_POD_RUNNING_WITH_SCISSORS_IN_THE_MATRIX_ARENA = (
        "running_with_scissors_in_the_matrix__arena"
    )
    MELTING_POD_TERRITORY_ROOMS = "territory__rooms"
    MELTING_POD_FRUIT_MARKET_CONCENTRIC_RIVERS = "fruit_market__concentric_rivers"
    MELTING_POD_PREDATOR_PREY_OPEN = "predator_prey__open"
    MELTING_POD_BOAT_RACE_EIGHT_RACES = "boat_race__eight_races"
    MELTING_POD_CHEMISTRY_THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS = (
        "chemistry__three_metabolic_cycles_with_plentiful_distractors"
    )
    MELTING_POD_CHICKEN_IN_THE_MATRIX_ARENA = "chicken_in_the_matrix__arena"
    MELTING_POD_COLLABORATIVE_COOKING_RING = "collaborative_cooking__ring"
    MELTING_POD_EXTERNALITY_MUSHROOMS_DENSE = "externality_mushrooms__dense"
    MELTING_POD_COMMONS_HARVEST_OPEN = "commons_harvest__open"
