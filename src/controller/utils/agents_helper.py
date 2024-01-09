import logging
import os
from os.path import join
from typing import Tuple, Type

from src.cfg_manager import get_cfg
from src.config.ctrl_config import (
    ACConfig,
    CtrlConfig,
    DemoMaConfig,
    GiftingConfig,
    LolaPGConfig,
    MaACConfig,
    MaConfig,
    MaMATEConfig,
    MateConfig,
)
from src.controller.actor_critic import ActorCritic
from src.controller.env_specific.coin_game.ma_coin_dont_take_others_coins import (
    MaCoinDontTakeOthersCoins,
)
from src.controller.env_specific.coin_game.ma_coin_to_middle import MaCoinToMiddle
from src.controller.env_specific.demo_ma import DemoMa
from src.controller.env_specific.ipd.punish_defect import MaIpdPunishDefect
from src.controller.env_specific.ma_social_wellfare import MaSocialWellfare
from src.controller.gifting import Gifting
from src.controller.lola_pg import LolaPG
from src.controller.mate import Mate
from src.controller.random_agents import RandomController
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
    if agent_type == AgentType.RANDOM:
        return RandomController, CtrlConfig
    elif agent_type == AgentType.ACTOR_CRITIC:
        return ActorCritic, ACConfig
    elif agent_type == AgentType.MATE:
        return Mate, MateConfig
    elif agent_type == AgentType.GIFTING:
        return Gifting, GiftingConfig
    elif agent_type == AgentType.LOLA_PG:
        return LolaPG, LolaPGConfig
    elif agent_type == AgentType.DEMO_MANIPULATION_AGENT:
        return DemoMa, DemoMaConfig
    elif agent_type == AgentType.MA_COIN_TO_MIDDLE_AC:
        return MaCoinToMiddle, MaACConfig
    elif agent_type == AgentType.MA_COIN_TO_MIDDLE_MATE:
        return MaCoinToMiddle, MaMATEConfig
    elif agent_type == AgentType.MA_IPD_PUNISH_DEFECT:
        return MaIpdPunishDefect, MaConfig
    elif agent_type == AgentType.MA_SOCIAL_WELLFARE_AC:
        return MaSocialWellfare, MaACConfig
    elif agent_type == AgentType.MA_SOCIAL_WELLFARE_MATE:
        return MaSocialWellfare, MaMATEConfig
    elif agent_type == AgentType.MA_COIN_DONT_TAKE_OTHERS_AC:
        return MaCoinDontTakeOthersCoins, MaACConfig
    elif agent_type == AgentType.MA_COIN_DONT_TAKE_OTHERS_MATE:
        return MaCoinDontTakeOthersCoins, MaMATEConfig

    raise NotImplementedError
