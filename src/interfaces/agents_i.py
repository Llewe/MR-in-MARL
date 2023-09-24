from abc import ABC, abstractmethod
from typing import TypeVar

from pettingzoo.utils.env import ActionType, ObsType, AgentID, ObsDict, ActionDict
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Space

from src.config.ctrl_configs.ctrl_config import CtrlConfig

C = TypeVar("C", bound=CtrlConfig)


class IAgents(ABC):
    @abstractmethod
    def __init__(self, config: C):
        pass

    @abstractmethod
    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ):
        pass

    @abstractmethod
    def init_new_episode(self):
        pass

    @abstractmethod
    def act(
        self,
        agent_id: AgentID,
        observation: ObsType,
        explore: bool = True,
    ) -> ActionType:
        pass

    @abstractmethod
    def update(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        curr_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def set_logger(self, writer: SummaryWriter) -> None:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass
