from abc import ABC

from gymnasium.spaces import Space
from pettingzoo.utils.env import AgentID

from src.config.ctrl_configs import actor_critic_config
from src.agents.implementations.utils.policy_network import PolicyNetwork
from src.agents.implementations.utils.state_value_network import StateValueNetwork
from src.interfaces import IAgents


class IAgentsGym(IAgents, ABC):
    policy_networks: dict[AgentID, PolicyNetwork]
    critic_networks: dict[AgentID, StateValueNetwork]

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ):
        self.policy_networks = {
            agent_id: PolicyNetwork(
                observation_space[agent_id].shape[0],
                action_space[agent_id].n,
                actor_critic_config.ACTOR_HIDDEN_UNITS,
            )
            for agent_id in action_space
        }

        self.critic_networks = {
            agent_id: StateValueNetwork(
                observation_space[agent_id].shape[0],
                actor_critic_config.CRITIC_HIDDEN_UNITS,
            )
            for agent_id in observation_space
        }

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
