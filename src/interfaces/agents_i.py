from abc import ABC, abstractmethod

from pettingzoo.utils.env import ActionType, ObsType, AgentID, ObsDict, ActionDict
from torch.utils.tensorboard import SummaryWriter
from gymnasium.spaces import Space


class IAgents(ABC):
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
        gamma: float,
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
