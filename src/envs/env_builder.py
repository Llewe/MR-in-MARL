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
    HarvestConfig,
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
    _setup_simple,
    _setup_simple_adversary,
    _setup_simple_spread,
    _setup_simple_tag,
)
from src.envs.setup.aec_envs import (
    _setup_chicken,
    _setup_my_coin_game,
    _setup_prisoners,
    _setup_samaritans,
    _setup_stag_hunt,
)
from src.envs.setup.parallel_envs import _p_setup_harvest, _p_setup_my_coin_game
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

        case EnvType.P_MY_COIN_GAME:
            return _p_setup_my_coin_game(writer, config)

        case EnvType.PRISONERS_DILEMMA:
            return _setup_prisoners(writer, config)

        case EnvType.SAMARITANS_DILEMMA:
            return _setup_samaritans(writer, config)
        case EnvType.STAG_HUNT:
            return _setup_stag_hunt(writer, config)
        case EnvType.CHICKEN:
            return _setup_chicken(writer, config)

        case EnvType.P_HARVEST:
            return _p_setup_harvest(writer, config)

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
        case EnvType.P_MY_COIN_GAME:
            return CoinGameConfig
        case EnvType.PRISONERS_DILEMMA:
            return PrisonersConfig
        case EnvType.SAMARITANS_DILEMMA:
            return SamaritansConfig
        case EnvType.STAG_HUNT:
            return StagHuntConfig
        case EnvType.CHICKEN:
            return ChickenConfig
        case EnvType.P_HARVEST:
            return HarvestConfig
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
