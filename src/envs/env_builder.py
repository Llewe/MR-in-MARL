import logging
import os
from os.path import join
from typing import Union, TypeVar, Type

from pettingzoo import AECEnv, ParallelEnv
from torch.utils.tensorboard import SummaryWriter

from src.config.env_config import (
    EnvConfig,
    SimpleTagConfig,
    SimpleSpreadConfig,
    SimpleAdversaryConfig,
    CoinGameConfig,
)
from src.enums.env_type_e import EnvType
from src.envs.setup import (
    _setup_simple,
    _setup_simple_tag,
    _setup_simple_spread,
    _setup_simple_adversary,
    _setup_parallel_simple,
    _setup_parallel_simple_tag,
    _setup_parallel_simple_spread,
    _setup_coin_game,
)
from src.utils.data_loader import load_pydantic_object, save_pydantic_object
from src.utils.utils import get_env_dir

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
    raise NotImplementedError


def get_env_class(env_name: EnvType) -> Type[T]:
    match env_name:
        case EnvType.SIMPLE:
            return EnvConfig
        case EnvType.SIMPLE_TAG:
            return SimpleTagConfig
        case EnvType.SIMPLE_SPREAD:
            return SimpleSpreadConfig
        case EnvType.SIMPLE_ADVERSARY:
            return SimpleAdversaryConfig
        case EnvType.COIN_GAME:
            return CoinGameConfig
    raise NotImplementedError


def load_env_config(env_name: EnvType) -> T:
    data_dir: str = get_env_dir()
    config_file: str = join(data_dir, "env_config.yml")

    c: Type[T] = get_env_class(env_name)

    if os.path.isfile(config_file):
        return load_pydantic_object(config_file, c)
    else:
        config = c()
        save_pydantic_object(config_file, config)
        return config
