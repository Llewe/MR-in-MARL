import logging
from typing import Union
from pettingzoo import AECEnv, ParallelEnv
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


def build_env(env_name: EnvType) -> Union[AECEnv, ParallelEnv]:
    logging.info(f"Building env: {env_name}")
    match env_name:
        case EnvType.SIMPLE:
            return _setup_simple()
        case EnvType.SIMPLE_TAG:
            return _setup_simple_tag()
        case EnvType.SIMPLE_SPREAD:
            return _setup_simple_spread()
        case EnvType.SIMPLE_ADVERSARY:
            return _setup_simple_adversary()
        case EnvType.P_SIMPLE:
            return _setup_parallel_simple()
        case EnvType.P_SIMPLE_TAG:
            return _setup_parallel_simple_tag()
        case EnvType.P_SIMPLE_SPREAD:
            return _setup_parallel_simple_spread()
        case EnvType.P_SIMPLE_ADVERSARY:
            raise NotImplementedError

        case EnvType.COIN_GAME:
            return _setup_coin_game()
    raise NotImplementedError
