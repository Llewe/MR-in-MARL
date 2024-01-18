from abc import ABC, abstractmethod
from typing import Dict, TypeVar

from gymnasium.spaces import Space
from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.enums.manipulation_modes_e import ManipulationMode
from src.enums.metrics_e import MetricsE
from src.interfaces import IController

BaseController = TypeVar("BaseController", bound=IController)


class IMaController(ABC):
    mode: ManipulationMode

    @abstractmethod
    def set_agents(
        self, agents: list[AgentID], observation_space: dict[AgentID, Space] | Space
    ) -> None:
        """
        Parameters
        ----------
        agents
        observation_space

        Returns
        -------

        """

    @abstractmethod
    def update_rewards(
        self,
        obs: ObsType | Dict[AgentID, ObsType],
        rewards: Dict[AgentID, float],
        metrics: Dict[MetricsE, float] | None,
    ) -> Dict[AgentID, float]:
        """
        Parameters
        ----------
        metrics
        ma_agents
        obs
        rewards

        Returns
        -------

        """

    @abstractmethod
    def episode_started(self, episode: int) -> None:
        """

        Parameters
        ----------
        episode

        Returns
        -------

        """

    @abstractmethod
    def episode_finished(self, episode: int, tag: str) -> None:
        """

        Parameters
        ----------
        episode
        tag

        Returns
        -------

        """

    @abstractmethod
    def epoch_started(self, epoch: int) -> None:
        """

        Parameters
        ----------
        epoch

        Returns
        -------

        """

    @abstractmethod
    def epoch_finished(self, epoch: int, tag: str) -> None:
        """

        Parameters
        ----------
        epoch
        tag

        Returns
        -------

        """

    @abstractmethod
    def set_logger(self, writer: SummaryWriter) -> None:
        pass

    @staticmethod
    def distribute_to_others(
        original_rewards: dict[AgentID, float],
        changed_rewards: dict[AgentID, float],
        punish_agent_id: AgentID,
        value: float,
        is_percentage: bool = True,
    ) -> float:
        punishment: float

        if is_percentage:
            punishment = original_rewards[punish_agent_id] * value
        else:
            punishment = value

        add_to_others = punishment / (len(original_rewards) - 1)

        for agent_id, reward in original_rewards.items():
            if agent_id == punish_agent_id:
                changed_rewards[agent_id] -= punishment
            else:
                changed_rewards[agent_id] += add_to_others

        return punishment
