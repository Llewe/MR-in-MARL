from typing import Generic, Optional, Type, TypeVar

from gymnasium import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.interfaces import IController

BaseController = TypeVar("BaseController", bound=IController)


class GenericHelper(Generic[BaseController], IController):
    """
    not the biggest fan of this currently, but I didn't find another way to make this
    TypeVar with bounds work in generics in python
    """

    base_class: BaseController

    def __init__(self, base_class: Type[BaseController], *args, **kwargs):
        self.base_class = base_class(*args, **kwargs)

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        self.base_class.init_agents(action_space, observation_space)

    def episode_started(self, episode: int) -> None:
        self.base_class.episode_started(episode)

    def episode_finished(self, episode: int, tag: str) -> None:
        self.base_class.episode_finished(episode, tag)

    def epoch_started(self, epoch: int) -> None:
        self.base_class.epoch_started(epoch)

    def epoch_finished(self, epoch: int, tag: str) -> None:
        self.base_class.epoch_finished(epoch, tag)

    def act(
        self, agent_id: AgentID, observation: ObsType, explore: bool = True
    ) -> ActionType:
        return self.base_class.act(agent_id, observation, explore)

    def step_agent(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        self.base_class.step_agent(
            agent_id, last_observation, last_action, reward, done
        )

    def step_finished(
        self, step: int, next_observations: Optional[dict[AgentID, ObsType]] = None
    ) -> None:
        self.base_class.step_finished(step, next_observations)

    def set_logger(self, writer: SummaryWriter) -> None:
        self.base_class.set_logger(writer)

    def save(self, path: str) -> None:
        self.base_class.save(path)

    def load(self, path: str) -> None:
        self.base_class.load(path)
