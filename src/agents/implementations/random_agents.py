from gymnasium import Space
from pettingzoo.utils.env import ObsDict, ActionDict, AgentID, ObsType, ActionType

from agents.agents_i import IAgents


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

    def act(self, agent_id: AgentID, observation: ObsType) -> (ActionType, float):
        return self.action_space[agent_id].sample(), 1.0

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
        # Random agents don't learn
        pass

    def act_parallel(self, observations: ObsDict) -> (ActionDict, dict[AgentID:float]):
        return {
            agent_id: self.action_space[agent_id].sample()
            for agent_id in self.action_space
        }

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
        # Random agents don't learn
        pass

    def save(self, path: str) -> None:
        # useless to save random agents
        pass

    def load(self, path: str) -> None:
        # useless to load random agents
        pass
