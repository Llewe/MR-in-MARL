import collections
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, List, Optional, Type, TypeVar, Generic

from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.controller.utils.generic_helper import GenericHelper, BaseController
from src.controller.actor_critic import ActorCritic
from gymnasium.spaces import Space
from src.interfaces import IController


@dataclass
class RewardHistory:
    reward_env: float
    reward_after_man: float
    manipulated_by_agent: dict[AgentID, float]
    manipulation_debts: float


class BaseRMP(Generic[BaseController], GenericHelper[BaseController]):
    """
    Manipulating rewards agent
    This agent controller is used to manipulate the rewards of other agents

    Extend your class with this AC implementation and add the callbacks for each
    manipulating agent inside the init function.
    Remember to call super().__init__(self, *args, **kwargs)

    """

    callbacks: dict[AgentID, Callable[[AgentID, ObsType, ActionType, float], float]]
    next_reward_offset_dict: dict[AgentID, float]
    reward_histories: dict[AgentID, list[RewardHistory]]
    agents: List[AgentID]

    def __init__(self, base_controller: Type[BaseController], *args, **kwargs):
        super().__init__(base_controller, *args, **kwargs)
        self.callbacks = {}

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        super().init_agents(action_space, observation_space)

        self.agents = []
        self.next_reward_offset_dict = {}
        self.reward_histories = {}

        for a in action_space.keys():
            self.agents.append(a)

    def reset_buffers(self) -> None:
        for a in self.agents:
            self.next_reward_offset_dict[a] = 0
            self.reward_histories[a] = []

    def set_callback(
        self,
        agent_id: AgentID,
        callback: Callable[[AgentID, ObsType, ActionType, float], float],
    ) -> None:
        """
        Set the callback for the given agent id
        Parameters
        ----------
        agent_id: AgentID
            The agent id for which the callback should be set. This agent is the
            manipulating agent
        callback: Callable[[AgentID, ObsType, ActionType, float], float]
            The callback function which should be called for the given agent id
            - AgentID -> Target agent id
            - ObsType -> Last observation of the target agent
            - ActionType -> Last action of the target agent
            - float -> Last reward of the target agent

            The callback function should return the manipulated reward offset as a float
             value
            The Target will get the reward + offset as reward
            and the caller of the callback will get the reward - offset as reward

        Returns
        -------


        """
        self.callbacks[agent_id] = callback

    def step_agent(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        manipulated_by_agent: dict[AgentID, float] = {}
        for a in self.agents:
            manipulated_by_agent[a] = 0

        # loop throw all callbacks and update the rewards individually
        for a_callback_id in self.callbacks:
            # skip if the agent is "itself"
            if a_callback_id == agent_id:
                continue

            reward_offset = self.callbacks[a_callback_id](
                agent_id, last_observation, last_action, reward
            )

            self.next_reward_offset_dict[a_callback_id] -= reward_offset
            self.next_reward_offset_dict[agent_id] += reward_offset

        # override the reward with the accumulated reward offset
        manipulated_reward = reward + self.next_reward_offset_dict[agent_id]
        self.next_reward_offset_dict[agent_id] = 0

        self.reward_histories[agent_id].append(
            RewardHistory(
                reward_env=reward,
                reward_after_man=manipulated_reward,
                manipulated_by_agent=manipulated_by_agent,
                manipulation_debts=0,
            )
        )

        super().step_agent(
            agent_id=agent_id,
            last_observation=last_observation,
            last_action=last_action,
            reward=manipulated_reward,
            done=done,
        )

    def epoch_started(self, epoch: int) -> None:
        self.reset_buffers()
        super().epoch_started(epoch)
