from typing import Callable
from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID, ObsType

from src.enums import EnvType
from src.enums.manipulation_modes_e import ManipulationMode
from src.envs.coin_game.coin_game import CoinGame
from src.envs.harvest.harvest import Harvest
from src.interfaces.ma_controller_i import IMaController
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

from gymnasium import Space


def get_ma_controller(
    manipulation_mode: ManipulationMode, env_type: EnvType
) -> IMaController:
    if manipulation_mode == ManipulationMode.NONE:
        from src.controller_ma.no_ma_controller import NoMaController

        return NoMaController()
    if manipulation_mode == ManipulationMode.CENTRAL_FIXED_PERCENTAGE:
        from src.controller_ma.central_ma_fix_percentage import CentralMaFixedPercentage

        return CentralMaFixedPercentage()

    if manipulation_mode == ManipulationMode.CENTRAL_AC_PERCENTAGE:
        from src.controller_ma.central_ma_ac_percentage import CentralMaAcPercentage

        return CentralMaAcPercentage()
    if (
        manipulation_mode == ManipulationMode.CENTRAL_HEURISTIC
        and env_type == EnvType.P_PRISONERS_DILEMMA
    ):
        from src.controller_ma.central_ma_heuristic_ipd import CentralMaHeuristicIPD

        return CentralMaHeuristicIPD()
    if manipulation_mode == ManipulationMode.INDIVIDUAL_ACTOR_CRITIC:
        from src.controller_ma.individual_ma_ac_percentage import (
            IndividualMaAcPercentage,
        )

        return IndividualMaAcPercentage()

    raise ValueError(f"Unknown manipulation mode: {manipulation_mode}")


def get_global_obs(
    manipulation_mode: ManipulationMode, env: ParallelEnv
) -> Callable[[], ObsType] | None:
    if (
        manipulation_mode == ManipulationMode.CENTRAL_HEURISTIC
        or manipulation_mode == ManipulationMode.CENTRAL_FIXED_PERCENTAGE
        or manipulation_mode == ManipulationMode.CENTRAL_AC_PERCENTAGE
    ):
        if isinstance(env, CoinGame):
            return env.get_global_obs
        if isinstance(
            env, aec_to_parallel_wrapper
        ):  # TODO: this is currently for the Dilemma env -> better would be a propper interface
            return env.aec_env.env.get_global_obs
        if isinstance(env, Harvest):
            return env.get_global_obs
        raise ValueError(
            f"Unknown env type: {type(env)}, don't know how to get global obs"
        )
    else:
        return None


def get_obs_space(
    manipulation_mode: ManipulationMode, env: ParallelEnv
) -> Space | dict[AgentID, Space]:
    if manipulation_mode == ManipulationMode.INDIVIDUAL_ACTOR_CRITIC:
        return {a: env.observation_space(a) for a in env.possible_agents}
    if (
        manipulation_mode == ManipulationMode.CENTRAL_AC_PERCENTAGE
        or manipulation_mode == ManipulationMode.CENTRAL_FIXED_PERCENTAGE
    ):
        if isinstance(env, Harvest):
            return env.get_global_obs_space()

    return env.observation_space(env.possible_agents[0])
