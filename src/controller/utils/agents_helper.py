import logging
import os
from os.path import join
from typing import Tuple, Type

from src.controller.actor_critic import ActorCritic
from src.controller.env_specific.coin_game.ma_coin_to_middle import MaCoinToMiddle
from src.controller.env_specific.demo_ma import DemoMa
from src.controller.mate import Mate
from src.controller.random_agents import RandomController
from src.cfg_manager import get_cfg

from src.controller.gifting import Gifting

from src.controller.env_specific.ipd.punish_defect import MaIpdPunishDefect
from src.controller.lola_pg import LolaPG
from src.config.ctrl_config import (
    ACConfig,
    CtrlConfig,
    DemoMaCoinConfig,
    DemoMaConfig,
    GiftingConfig,
    LolaPGConfig,
    MateConfig,
    MaConfig,
)
from src.enums.agent_type_e import AgentType
from src.interfaces.controller_i import IController
from src.utils.data_loader import load_pydantic_object, save_pydantic_object


def get_agents() -> IController:
    logging.info(f"Getting agents for {get_cfg().exp_config.AGENT_TYPE}")
    type_class, config_class = get_agent_class(get_cfg().exp_config.AGENT_TYPE)

    config: CtrlConfig = load_ctrl_config(config_class)

    return type_class(config)


def load_ctrl_config(config_class: Type[CtrlConfig]) -> CtrlConfig:
    data_dir: str = get_cfg().get_ctrl_dir()
    config_file: str = join(data_dir, "controller.json")

    if os.path.isfile(config_file):
        return load_pydantic_object(config_file, config_class)
    else:
        config = config_class()
        save_pydantic_object(config_file, config)
        return config


def get_agent_class(
    agent_type: AgentType,
) -> Tuple[Type[IController], Type[CtrlConfig]]:
    match agent_type:
        case AgentType.RANDOM:
            return RandomController, CtrlConfig
        case AgentType.ACTOR_CRITIC:
            return ActorCritic, ACConfig

        case AgentType.MATE:
            return Mate, MateConfig

        case AgentType.GIFTING:
            return Gifting, GiftingConfig

        case AgentType.LOLA_PG:
            return LolaPG, LolaPGConfig

        case AgentType.DEMO_MANIPULATION_AGENT:
            return DemoMa, DemoMaConfig

        case AgentType.MA_COIN_TO_MIDDLE:
            return MaCoinToMiddle, MaConfig

        case AgentType.MA_IPD_PUNISH_DEFECT:
            return MaIpdPunishDefect, MaConfig

    raise NotImplementedError