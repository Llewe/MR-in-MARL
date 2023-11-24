import logging
import os
from os.path import join
from typing import Type, TypeVar, Union

from pettingzoo import AECEnv, ParallelEnv
from torch.utils.tensorboard import SummaryWriter

from src.cfg_manager import get_cfg
from src.config.env_config import (
    ChickenConfig,
    CoinGameConfig,
    EnvConfig,
    PrisonersConfig,
    SamaritansConfig,
    SimpleAdversaryConfig,
    SimpleSpreadConfig,
    SimpleTagConfig,
    StagHuntConfig,
)
from src.enums.env_type_e import EnvType
from src.envs.setup import (
    _setup_coin_game,
    _setup_parallel_simple,
    _setup_parallel_simple_spread,
    _setup_parallel_simple_tag,
    _setup_simple,
    _setup_simple_adversary,
    _setup_simple_spread,
    _setup_simple_tag,
)
from src.envs.setup.aec_envs import (
    _setup_chicken,
    _setup_melting_pod,
    _setup_my_coin_game,
    _setup_prisoners,
    _setup_samaritans,
    _setup_stag_hunt,
)
from src.utils.data_loader import load_pydantic_object, save_pydantic_object

T = TypeVar("T", bound=EnvConfig)


def build_env(
    env_name: EnvType, writer: SummaryWriter = None
) -> Union[AECEnv, ParallelEnv]:
    logging.info(f"Building env: {env_name}")
    config: T = load_env_config(env_name)

    match env_name:
        case EnvType.SIMPLE:
            return _setup_simple(config)
        case EnvType.SIMPLE_TAG:
            return _setup_simple_tag(config)
        case EnvType.SIMPLE_SPREAD:
            return _setup_simple_spread(config)
        case EnvType.SIMPLE_ADVERSARY:
            return _setup_simple_adversary(config)
        case EnvType.P_SIMPLE:
            return _setup_parallel_simple()
        case EnvType.P_SIMPLE_TAG:
            return _setup_parallel_simple_tag()
        case EnvType.P_SIMPLE_SPREAD:
            return _setup_parallel_simple_spread()
        case EnvType.P_SIMPLE_ADVERSARY:
            raise NotImplementedError

        case EnvType.COIN_GAME:
            return _setup_coin_game(writer, config)

        case EnvType.MY_COIN_GAME:
            return _setup_my_coin_game(writer, config)

        case EnvType.PRISONERS_DILEMMA:
            return _setup_prisoners(writer, config)

        case EnvType.SAMARITANS_DILEMMA:
            return _setup_samaritans(writer, config)
        case EnvType.STAG_HUNT:
            return _setup_stag_hunt(writer, config)
        case EnvType.CHICKEN:
            return _setup_chicken(writer, config)
        case (
            EnvType.MELTING_POD_PRISONERS_DILEMMA_IN_THE_MATRIX_ARENA
            | EnvType.MELTING_POD_STAG_HUNT_IN_THE_MATRIX_ARENA
            | EnvType.MELTING_POD_COLLABORATIVE_COOKING_ASYMMETRIC
            | EnvType.MELTING_POD_PURE_COORDINATION_IN_THE_MATRIX_ARENA
            | EnvType.MELTING_POD_COINS
            | EnvType.MELTING_POD_HIDDEN_AGENDA
            | EnvType.MELTING_POD_PREDATOR_PREY_RANDOM_FOREST
            | EnvType.MELTING_POD_CHEMISTRY_TWO_METABOLIC_CYCLES_WITH_DISTRACTORS
            | EnvType.MELTING_POD_RATIONALIZABLE_COORDINATION_IN_THE_MATRIX_ARENA
            | EnvType.MELTING_POD_DAYCARE
            | EnvType.MELTING_POD_RUNNING_WITH_SCISSORS_IN_THE_MATRIX_REPEATED
            | EnvType.MELTING_POD_BACH_OR_STRAVINSKY_IN_THE_MATRIX_REPEATED
            | EnvType.MELTING_POD_PURE_COORDINATION_IN_THE_MATRIX_REPEATED
            | EnvType.MELTING_POD_COLLABORATIVE_COOKING_CRAMPED
            | EnvType.MELTING_POD_COOP_MINING
            | EnvType.MELTING_POD_COMMONS_HARVEST_CLOSED
            | EnvType.MELTING_POD_COLLABORATIVE_COOKING_FORCED
            | EnvType.MELTING_POD_PRISONERS_DILEMMA_IN_THE_MATRIX_REPEATED
            | EnvType.MELTING_POD_FACTORY_COMMONS_EITHER_OR
            | EnvType.MELTING_POD_CHEMISTRY_TWO_METABOLIC_CYCLES
            | EnvType.MELTING_POD_TERRITORY_OPEN
            | EnvType.MELTING_POD_GIFT_REFINEMENTS
            | EnvType.MELTING_POD_ALLELOPATHIC_HARVEST_OPEN
            | EnvType.MELTING_POD_CHEMISTRY_THREE_METABOLIC_CYCLES
            | EnvType.MELTING_POD_STAG_HUNT_IN_THE_MATRIX_REPEATED
            | EnvType.MELTING_POD_PREDATOR_PREY_ORCHARD
            | EnvType.MELTING_POD_BACH_OR_STRAVINSKY_IN_THE_MATRIX_ARENA
            | EnvType.MELTING_POD_PAINTBALL_KING_OF_THE_HILL
            | EnvType.MELTING_POD_PREDATOR_PREY_ALLEY_HUNT
            | EnvType.MELTING_POD_TERRITORY_INSIDE_OUT
            | EnvType.MELTING_POD_CHICKEN_IN_THE_MATRIX_REPEATED
            | EnvType.MELTING_POD_RUNNING_WITH_SCISSORS_IN_THE_MATRIX_ONE_SHOT
            | EnvType.MELTING_POD_PAINTBALL_CAPTURE_THE_FLAG
            | EnvType.MELTING_POD_COLLABORATIVE_COOKING_CIRCUIT
            | EnvType.MELTING_POD_RATIONALIZABLE_COORDINATION_IN_THE_MATRIX_REPEATED
            | EnvType.MELTING_POD_COLLABORATIVE_COOKING_FIGURE_EIGHT
            | EnvType.MELTING_POD_CLEAN_UP
            | EnvType.MELTING_POD_COMMONS_HARVEST_PARTNERSHIP
            | EnvType.MELTING_POD_COLLABORATIVE_COOKING_CROWDED
            | EnvType.MELTING_POD_RUNNING_WITH_SCISSORS_IN_THE_MATRIX_ARENA
            | EnvType.MELTING_POD_TERRITORY_ROOMS
            | EnvType.MELTING_POD_FRUIT_MARKET_CONCENTRIC_RIVERS
            | EnvType.MELTING_POD_PREDATOR_PREY_OPEN
            | EnvType.MELTING_POD_BOAT_RACE_EIGHT_RACES
            | EnvType.MELTING_POD_CHEMISTRY_THREE_METABOLIC_CYCLES_WITH_PLENTIFUL_DISTRACTORS
            | EnvType.MELTING_POD_CHICKEN_IN_THE_MATRIX_ARENA
            | EnvType.MELTING_POD_COLLABORATIVE_COOKING_RING
            | EnvType.MELTING_POD_EXTERNALITY_MUSHROOMS_DENSE
            | EnvType.MELTING_POD_COMMONS_HARVEST_OPEN
        ):
            return _setup_melting_pod(env_name.value, config)

    raise NotImplementedError


def get_env_class(env_name: EnvType) -> Type[T]:
    match env_name:
        case EnvType.SIMPLE_TAG:
            return SimpleTagConfig
        case EnvType.SIMPLE_SPREAD:
            return SimpleSpreadConfig
        case EnvType.SIMPLE_ADVERSARY:
            return SimpleAdversaryConfig
        case EnvType.COIN_GAME:
            return CoinGameConfig
        case EnvType.MY_COIN_GAME:
            return CoinGameConfig
        case EnvType.PRISONERS_DILEMMA:
            return PrisonersConfig
        case EnvType.SAMARITANS_DILEMMA:
            return SamaritansConfig
        case EnvType.STAG_HUNT:
            return StagHuntConfig
        case EnvType.CHICKEN:
            return ChickenConfig
        case _:
            return EnvConfig


def load_env_config(env_name: EnvType) -> T:
    data_dir: str = get_cfg().get_env_dir()
    config_file: str = join(data_dir, "env_config.yml")

    c: Type[T] = get_env_class(env_name)

    if os.path.isfile(config_file):
        return load_pydantic_object(config_file, c)
    else:
        config = c()
        save_pydantic_object(config_file, config)
        return config
