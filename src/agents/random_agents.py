from gymnasium import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.config.ctrl_config import CtrlConfig
from src.interfaces.agents_i import IAgents


class RandomAgents(IAgents):
    def __init__(self, config: CtrlConfig):
        pass

    action_space: dict[AgentID, Space]
    observation_space: dict[AgentID, Space]

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    def epoch_started(self, epoch: int):
        # Nothing interesting to do here
        pass

    def epoch_finished(self, epoch: int):
        # Nothing interesting to do here
        pass

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        return self.action_space[agent_id].sample()

    def update(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        # Random agents don't learn
        pass

    def save(self, path: str) -> None:
        # useless to save random agents
        pass

    def load(self, path: str) -> None:
        # useless to load random agents
        pass

    def set_logger(self, writer: SummaryWriter) -> None:
        # Random agents don't learn
        pass
