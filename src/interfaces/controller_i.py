from abc import ABC, abstractmethod
from typing import Optional

from gymnasium.spaces import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.config.ctrl_config import CtrlConfig


class IController(ABC):
    @abstractmethod
    def __init__(self, config: CtrlConfig):
        """
        Parameters
        ----------
        config
        """

    @abstractmethod
    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        """
        Parameters
        ----------
        action_space
        observation_space

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
    def act(
        self,
        agent_id: AgentID,
        observation: ObsType,
        explore: bool = True,
    ) -> ActionType:
        """

        Parameters
        ----------
        agent_id
        observation
        explore

        Returns
        -------

        """

    def act_parallel(
        self,
        observations: dict[AgentID, ObsType],
        explore: bool = True,
    ) -> dict[AgentID, ActionType]:
        return {
            agent_id: self.act(agent_id, observation, explore)
            for agent_id, observation in observations.items()
        }

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

    def step_agent_parallel(
        self,
        last_observations: dict[AgentID, ObsType],
        last_actions: dict[AgentID, ActionType],
        rewards: dict[AgentID, float],
        dones: dict[AgentID, bool],
    ) -> None:
        for agent_id in last_observations.keys():
            self.step_agent(
                agent_id,
                last_observations[agent_id],
                last_actions[agent_id],
                rewards[agent_id],
                dones[agent_id],
            )

    @abstractmethod
    def step_finished(
        self, step: int, next_observations: Optional[dict[AgentID, ObsType]] = None
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
