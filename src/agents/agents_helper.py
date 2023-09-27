import logging
import os
from os.path import join
from typing import TypeVar, Type

from src.config.base_ctrl_config import BaseCtrlConfig
from src.config.ctrl_configs.a2c_config import A2cConfig
from src.config.ctrl_configs.actor_critic_config import ActorCriticConfig
from src.config.ctrl_configs.ctrl_config import CtrlConfig
from src.config.ctrl_configs.demo_ma_config import DemoMaConfig
from src.enums.agent_type_e import AgentType
from src.interfaces.agents_i import IAgents, C
from src.utils.data_loader import load_pydantic_object, save_pydantic_object
from src.utils.utils import get_ctrl_dir
from .implementations import (
    RandomAgents,
    ActorCritic,
    DemoMa,
)
from .implementations.a2c import A2C

T = TypeVar("T", bound=IAgents)


def get_agents() -> IAgents:
    base_ctrl_config = BaseCtrlConfig()

    logging.info(f"Getting agents for {base_ctrl_config.AGENT_TYPE}")
    type_class, config_class = get_agent_class(base_ctrl_config.AGENT_TYPE)

    config: C = load_ctrl_config(config_class)

    return type_class(config)


def load_ctrl_config(config_class: Type[C]) -> C:
    data_dir: str = get_ctrl_dir()
    config_file: str = join(data_dir, "controller.yml")

    if os.path.isfile(config_file):
        return load_pydantic_object(config_file, config_class)
    else:
        config = config_class()
        save_pydantic_object(config_file, config)
        return config


def get_agent_class(agent_type: AgentType) -> (Type[T], Type[C]):
    match agent_type:
        case AgentType.RANDOM:
            return RandomAgents, CtrlConfig

        case AgentType.ACTOR_CRITIC:
            return ActorCritic, ActorCriticConfig

        case AgentType.DEMO_MANIPULATION_AGENT:
            return DemoMa, DemoMaConfig

        case AgentType.ACTOR_CRITIC_TD0:
            raise NotImplementedError

        case AgentType.ACTOR_CRITIC_TD_LAMBDA_BACKWARD_VIEW:
            raise NotImplementedError

        case AgentType.A2C:
            return A2C, A2cConfig

    raise NotImplementedError
