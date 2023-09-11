from typing import Union
from pettingzoo import AECEnv, ParallelEnv
from .enums.env_type_e import EnvType
from .aec_envs import (
    _setup_simple,
    _setup_simple_tag,
    _setup_simple_spread,
    _setup_simple_adversary,
)
from .parallel_envs import (
    _setup_parallel_simple,
    _setup_parallel_simple_tag,
    _setup_parallel_simple_spread,
)


def build_env(env_name: EnvType) -> Union[AECEnv, ParallelEnv]:
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
    raise NotImplementedError
