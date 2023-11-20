from typing import Callable

from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.agents.implementations.old.actor_critic import ActorCritic


class MRAgent(ActorCritic):
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
            The agent id for which the callback should be set
        callback: Callable[[AgentID, ObsType, ActionType, float], float]
            The callback function which should be called for the given agent id

            The callback function should return the manipulated reward offset as a float value

        Returns
        -------

        """
        self.mr_callbacks[agent_id] = callback

    def update(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        curr_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        # not the cleanest solution but it works for now
        if agent_id not in self.next_reward_offset_dict:
            self.next_reward_offset_dict[agent_id] = 0

        # loop throw all callbacks and update the rewards individually
        for a_callback_id in self.mr_callbacks:
            # skip if the agent is "itself"
            if a_callback_id == agent_id:
                continue

            # not the cleanest solution but it works for now
            if a_callback_id not in self.next_reward_offset_dict:
                self.next_reward_offset_dict[a_callback_id] = 0

            reward_offset = self.mr_callbacks[a_callback_id](
                agent_id, last_observation, last_action, reward
            )
            self.next_reward_offset_dict[a_callback_id] -= reward_offset
            self.next_reward_offset_dict[agent_id] += reward_offset

        # override the reward with the accumulated reward offset
        reward += self.next_reward_offset_dict[agent_id]
        self.next_reward_offset_dict[agent_id] = 0

        super().update(
            agent_id=agent_id,
            last_observation=last_observation,
            curr_observation=curr_observation,
            last_action=last_action,
            reward=reward,
            done=done,
        )
