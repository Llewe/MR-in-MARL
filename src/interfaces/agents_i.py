from abc import ABC, abstractmethod

from gymnasium.spaces import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.config.ctrl_config import CtrlConfig


class IAgents(ABC):
    @abstractmethod
    def __init__(self, config: CtrlConfig):
        pass

    @abstractmethod
    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        pass

    @abstractmethod
    def epoch_started(self, epoch: int) -> None:
        pass

    @abstractmethod
    def epoch_finished(self, epoch: int) -> None:
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
    def step_agent(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        pass

    @abstractmethod
    def step_finished(self, step: int) -> None:
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
