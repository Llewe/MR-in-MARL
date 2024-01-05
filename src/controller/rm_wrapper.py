import collections
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.controller.actor_critic import ActorCritic


@dataclass
class RewardHistory:
    reward_env: float
    reward_after_man: float
    manipulated_by_agent: dict[AgentID, float]
    manipulation_debts: float


class RmWrapper(ActorCritic):
    """
    Manipulating rewards agent
    This agent controller is used to manipulate the rewards of other agents

    Just extend your class with this AC implementation and add the callbacks for each
    manipulating agent inside the init function.
    Don't forget to call super().__init__(self, *args, **kwargs)

    """

    mr_callbacks: dict[
        AgentID, Callable[[AgentID, ObsType, ActionType, float], float]
    ] = {}

    next_reward_offset_dict: dict[AgentID, float] = {}

    reward_histories: dict[AgentID, list[RewardHistory]] = {}

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
            The agent id for which the callback should be set. This agent is the manipulating agent
        callback: Callable[[AgentID, ObsType, ActionType, float], float]
            The callback function which should be called for the given agent id
            - AgentID -> Target agent id
            - ObsType -> Last observation of the target agent
            - ActionType -> Last action of the target agent
            - float -> Last reward of the target agent

            The callback function should return the manipulated reward offset as a float value

        Returns
        -------

        """
        self.mr_callbacks[agent_id] = callback

    def step_agent(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        manipulated_by_agent: dict[AgentID, float] = {}
        for a in self.actor_networks:
            manipulated_by_agent[a] = 0


        # loop throw all callbacks and update the rewards individually
        for a_callback_id in self.mr_callbacks:
            # skip if the agent is "itself"
            if a_callback_id == agent_id:
                continue

            reward_offset = self.mr_callbacks[a_callback_id](
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
    def step_finished(
            self, step: int, next_observations: Optional[dict[AgentID, ObsType]] = None
    ) -> None:
        super().step_finished(step, next_observations)


    def epoch_started(self, epoch: int) -> None:
        super().epoch_started(epoch)

        self.reward_histories.clear()


        for a in self.actor_networks:
            self.next_reward_offset_dict[a] = 0

        for agent_id in self.next_reward_offset_dict:
            self.reward_histories[agent_id] = []

    def epoch_finished(self, epoch: int, tag: str) -> None:
        super().epoch_finished(epoch,tag)
