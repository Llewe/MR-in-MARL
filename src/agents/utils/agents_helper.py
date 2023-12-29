import logging
import os
from os.path import join
from typing import Tuple, Type

from src.agents.a2c import A2C
from src.agents.env_specific.coingame.demo_ma_coin import DemoMaCoin
from src.agents.env_specific.demo_ma import DemoMa
from src.agents.mate import Mate
from src.agents.random_agents import RandomAgents
from src.cfg_manager import get_cfg

from src.agents.gifting import Gifting
from src.config.ctrl_config import (
    A2cConfig,
    CtrlConfig,
    DemoMaCoinConfig,
    DemoMaConfig,
    GiftingConfig,
    MateConfig,
)
from src.enums.agent_type_e import AgentType
from src.interfaces.agents_i import IAgents
from src.utils.data_loader import load_pydantic_object, save_pydantic_object


def get_agents() -> IAgents:
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


def get_agent_class(agent_type: AgentType) -> Tuple[Type[IAgents], Type[CtrlConfig]]:
    match agent_type:
        case AgentType.RANDOM:
            return RandomAgents, CtrlConfig
        case AgentType.A2C:
            return A2C, A2cConfig

        case AgentType.MATE:
            return Mate, MateConfig

        case AgentType.GIFTING:
            return Gifting, GiftingConfig

        case AgentType.DEMO_MANIPULATION_AGENT:
            return DemoMa, DemoMaConfig

        case AgentType.DEMO_MANIPULATION_COIN:
            return DemoMaCoin, DemoMaCoinConfig

    raise NotImplementedError
