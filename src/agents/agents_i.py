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
    ) -> (ActionType, float):
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
        log_prob: float,
        gamma: float,
    ) -> None:
        pass

    @abstractmethod
    def act_parallel(self, observations: ObsDict) -> (ActionDict, dict[AgentID:float]):
        pass

    @abstractmethod
    def update_parallel(
        self,
        observations: ObsDict,
        next_observations: ObsDict,
        actions: ActionDict,
        rewards: dict[AgentID:float],
        dones: dict[AgentID:bool],
        log_probs: dict[AgentID:float],
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
