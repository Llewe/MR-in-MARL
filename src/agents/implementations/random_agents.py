from gymnasium import Space
from pettingzoo.utils.env import ObsDict, ActionDict, AgentID, ObsType, ActionType

from src.agents.agents_i import IAgents
from torch.utils.tensorboard import SummaryWriter


class RandomAgents(IAgents):
    action_space: dict[AgentID, Space]
    observation_space: dict[AgentID, Space]

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ):
        self.action_space = action_space
        self.observation_space = observation_space

    def init_new_episode(self):
        # Random agents don't learn
        pass

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        return self.action_space[agent_id].sample()

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
