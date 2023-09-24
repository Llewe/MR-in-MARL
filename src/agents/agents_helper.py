import logging
import os
from os.path import join
from typing import TypeVar, Type

from src import training_config
from src.enums.agent_type_e import AgentType
from src.interfaces.agents_i import IAgents, C
from .implementations import (
    RandomAgents,
    ActorCritic,
    DemoMa,
)
from .implementations.a2c import A2C
from .implementations.a2c_examples import (
    ActorCriticTd0,
    ActorCriticTdLamdaBack,
)
from src.config.ctrl_configs.actor_critic_config import ActorCriticConfig
from src.config.ctrl_configs.ctrl_config import CtrlConfig
from src.utils.data_loader import load_pydantic_object, save_pydantic_object
from src.utils.utils import get_ctrl_dir
from ..config.ctrl_configs.demo_ma_config import DemoMaConfig

T = TypeVar("T", bound=IAgents)


def get_agents(agent_type: AgentType = training_config.AGENT_TYPE) -> IAgents:
    logging.info(f"Getting agents for {agent_type}")
    type_class, config_class = get_agent_class(agent_type)

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
            return ActorCriticTd0, CtrlConfig

        case AgentType.ACTOR_CRITIC_TD_LAMBDA_BACKWARD_VIEW:
            return ActorCriticTdLamdaBack, CtrlConfig

        case AgentType.A2C:
            return A2C, CtrlConfig

    raise NotImplementedError
