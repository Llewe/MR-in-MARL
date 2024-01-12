from abc import ABC, abstractmethod
from typing import TypeVar

from gymnasium.spaces import Space
from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.enums.manipulation_modes_e import ManipulationMode
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
        self, obs: ObsType | dict[AgentID, ObsType], rewards: dict[AgentID, float]
    ) -> dict[AgentID, float]:
        """
        Parameters
        ----------
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
