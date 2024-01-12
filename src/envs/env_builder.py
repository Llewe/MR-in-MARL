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

from src.envs.env_setup_parallel import (
    _p_setup_harvest,
    _p_setup_coin_game,
    _p_setup_prisoners,
    _p_setup_simple_tag,
)
from src.envs.env_setup_aec import (
    _setup_chicken,
    _setup_coin_game,
    _setup_prisoners,
    _setup_samaritans,
    _setup_simple,
    _setup_simple_adversary,
    _setup_simple_spread,
    _setup_simple_tag,
    _setup_stag_hunt,
)

from src.utils.data_loader import load_pydantic_object, save_pydantic_object

T = TypeVar("T", bound=EnvConfig)


def build_env(
    env_name: EnvType, writer: SummaryWriter = None
) -> Union[AECEnv, ParallelEnv]:
    logging.info(f"Building env: {env_name}")
    config: T = load_env_config(env_name)

    if env_name == EnvType.SIMPLE:
        return _setup_simple(config)
    elif env_name == EnvType.SIMPLE_TAG:
        return _setup_simple_tag(config)
    elif env_name == EnvType.SIMPLE_SPREAD:
        return _setup_simple_spread(config)
    elif env_name == EnvType.SIMPLE_ADVERSARY:
        return _setup_simple_adversary(config)
    elif env_name == EnvType.P_SIMPLE:
        raise NotImplementedError
    elif env_name == EnvType.P_SIMPLE_TAG:
        return _p_setup_simple_tag(writer, config)
    elif env_name == EnvType.P_SIMPLE_SPREAD:
        raise NotImplementedError
    elif env_name == EnvType.P_SIMPLE_ADVERSARY:
        raise NotImplementedError
    elif env_name == EnvType.COIN_GAME:
        return _setup_coin_game(writer, config)
    elif env_name == EnvType.P_COIN_GAME:
        return _p_setup_coin_game(writer, config)
    elif env_name == EnvType.PRISONERS_DILEMMA:
        return _setup_prisoners(writer, config)
    elif env_name == EnvType.P_PRISONERS_DILEMMA:
        return _p_setup_prisoners(writer, config)
    elif env_name == EnvType.SAMARITANS_DILEMMA:
        return _setup_samaritans(writer, config)
    elif env_name == EnvType.STAG_HUNT:
        return _setup_stag_hunt(writer, config)
    elif env_name == EnvType.CHICKEN:
        return _setup_chicken(writer, config)
    elif env_name == EnvType.P_HARVEST:
        return _p_setup_harvest(writer, config)
    else:
        raise NotImplementedError


def get_env_class(env_name: EnvType) -> Type[T]:
    if env_name == EnvType.SIMPLE_TAG or env_name == EnvType.P_SIMPLE_TAG:
        return SimpleTagConfig
    elif env_name == EnvType.SIMPLE_SPREAD:
        return SimpleSpreadConfig
    elif env_name == EnvType.SIMPLE_ADVERSARY:
        return SimpleAdversaryConfig
    elif env_name == EnvType.COIN_GAME:
        return CoinGameConfig
    elif env_name == EnvType.P_COIN_GAME:
        return CoinGameConfig
    elif env_name == EnvType.PRISONERS_DILEMMA:
        return PrisonersConfig
    elif env_name == EnvType.SAMARITANS_DILEMMA:
        return SamaritansConfig
    elif env_name == EnvType.STAG_HUNT:
        return StagHuntConfig
    elif env_name == EnvType.CHICKEN:
        return ChickenConfig
    elif env_name == EnvType.P_HARVEST:
        return HarvestConfig
    else:
        return EnvConfig


def load_env_config(env_name: EnvType) -> T:
    data_dir: str = get_cfg().get_env_dir()
    config_file: str = join(data_dir, "env_config.json")

    c: Type[T] = get_env_class(env_name)

    if os.path.isfile(config_file):
        return load_pydantic_object(config_file, c)
    else:
        config = c()
        save_pydantic_object(config_file, config)
        return config
