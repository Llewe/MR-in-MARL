from statistics import mean
from typing import Optional

from gymnasium import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from src.controller.actor_critic import ActorCritic

from src.config.ctrl_config import GiftingConfig
from src.utils.gym_utils import (
    add_action_to_space,
    get_action_from_index,
    get_space_size,
)


class Gifting(ActorCritic):
    gift_mode: GiftingConfig.Mode
    gift_reward: float

    action_mode: GiftingConfig.ActionMode
    env_none_action_index: int

    original_action_space: dict[AgentID, Space]
    original_action_length: dict[AgentID, int]
    agent_index_to_id: dict[int, AgentID]

    gifts_in_transit: dict[AgentID, float]
    gift_received: dict[AgentID, float]

    stats_gift_send: dict[AgentID, float]
    stats_gift_received: dict[AgentID, float]

    current_budget: dict[AgentID, float]
    gift_init_budget: float


    steps: int

    def __init__(self, config: GiftingConfig):
        super(Gifting, self).__init__(config)

        self.gift_mode = config.GIFT_MODE
        self.gift_reward = config.GIFT_REWARD

        self.action_mode = config.ACTION_MODE
        self.env_none_action_index = config.ENV_NONE_ACTION_INDEX

        self.gift_init_budget = config.GIFT_BUDGET

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        self.agent_index_to_id = {}
        for i, agent_id in enumerate(action_space.keys()):
            self.agent_index_to_id[i] = agent_id

        self.original_action_space = {}
        self.original_action_length = {}

        self.stats_gift_send = {}
        self.stats_gift_received = {}

        self.gifts_in_transit = {}
        self.gift_received = {}

        self.current_budget = {}

        nr_extra_actions = (
            action_space.keys().__len__() - 1
        )  # -1 one can't gift to itself

        for agent_id, space in action_space.items():
            self.original_action_space[agent_id] = space
            self.original_action_length[agent_id] = get_space_size(space)
            action_space[agent_id] = add_action_to_space(space, nr_extra_actions)

        super(Gifting, self).init_agents(action_space, observation_space)

    def epoch_started(self, epoch: int) -> None:
        super(Gifting, self).epoch_started(epoch)

        for agent_id in self.agent_index_to_id.values():
            self.gifts_in_transit[agent_id] = 0.0
            self.gift_received[agent_id] = 0.0
            self.stats_gift_send[agent_id] = 0.0
            self.stats_gift_received[agent_id] = 0.0
            if self.gift_mode == GiftingConfig.Mode.FIXED_BUDGET:
                self.current_budget[agent_id] = self.gift_init_budget
            elif self.gift_mode == GiftingConfig.Mode.REPLENISHABLE_BUDGET:
                self.current_budget[agent_id] = 0.0

        self.steps = 0

    def epoch_finished(self, epoch: int, tag: str) -> None:
        super(Gifting, self).epoch_finished(epoch, tag)

        if self.writer is not None and self.steps > 0:
            self.writer.add_scalar(
                f"{tag}/gifting/gifts_send",
                mean(self.stats_gift_send.values()) / self.steps,
                epoch,
            )

            self.writer.add_scalar(
                f"{tag}/gifting/gifts_received",
                mean(self.stats_gift_received.values()) / self.steps,
                epoch,
            )
            if self.gift_mode == GiftingConfig.Mode.FIXED_BUDGET or self.gift_mode == GiftingConfig.Mode.REPLENISHABLE_BUDGET:
                self.writer.add_scalar(
                    f"{tag}/gifting/remaining_budget",
                    mean(self.current_budget.values()) / self.steps,
                    epoch,
                )


    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        action = super(Gifting, self).act(agent_id, observation, explore)

        if self.original_action_space[agent_id].contains(action):
            return action

        # Handle the gift action
        target_agent_id = self.get_target_agent(agent_id, action)
        if self.gift_mode == GiftingConfig.Mode.ZERO_SUM:
            self.send_gift_zero_sum(agent_id, target_agent_id)
        elif self.gift_mode == GiftingConfig.Mode.FIXED_BUDGET:
            self.send_gift_fixed_budget(agent_id, target_agent_id)
        elif self.gift_mode == GiftingConfig.Mode.REPLENISHABLE_BUDGET:
            self.send_gift_replenishable_budget(agent_id, target_agent_id)
        else:
            raise ValueError("Unsupported gift mode")

        # Select action for the environment
        if self.action_mode == GiftingConfig.ActionMode.RANDOM:
            return self.original_action_space[agent_id].sample()

        if self.action_mode == GiftingConfig.ActionMode.NO_ACTION:
            return get_action_from_index(
                self.original_action_space[agent_id], self.env_none_action_index
            )

        raise ValueError("Unsupported action mode")

    def step_agent(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:


        # recieve the gift
        if self.gift_mode == GiftingConfig.Mode.ZERO_SUM:
            reward += self.gift_received[agent_id]
        elif self.gift_mode == GiftingConfig.Mode.FIXED_BUDGET:
            # if an agent has no budget left, it can't receive gifts anymore
            if self.current_budget[agent_id] - self.gift_reward < 0 :
                self.gift_received[agent_id] = 0.0
            else:
                reward += self.gift_received[agent_id]
        elif self.gift_mode == GiftingConfig.Mode.REPLENISHABLE_BUDGET:
            budget_increase = reward*0.5

            self.current_budget[agent_id] += budget_increase
            if self.current_budget[agent_id] - self.gift_reward > 0 :
                reward += self.gift_received[agent_id]




        # handle learning by underlying algorithm
        super(Gifting, self).step_agent(
            agent_id, last_observation, last_action, reward, done
        )

    def get_target_agent(self, agent_id: AgentID, action: ActionType) -> AgentID:
        target_index = action - self.original_action_length[agent_id]
        return self.agent_index_to_id[target_index]

    def send_gift_zero_sum(self, agent_id: AgentID, target_agent_id: AgentID) -> None:
        self.gifts_in_transit[agent_id] -= self.gift_reward
        self.gifts_in_transit[target_agent_id] += self.gift_reward

        self.stats_gift_send[agent_id] += self.gift_reward

    def send_gift_fixed_budget(self, agent_id: AgentID, target_agent_id: AgentID) -> None:
        if self.current_budget[agent_id]-self.gift_reward >= 0:
            self.gifts_in_transit[agent_id] -= self.gift_reward
            self.gifts_in_transit[target_agent_id] += self.gift_reward

            self.stats_gift_send[agent_id] += self.gift_reward
            self.current_budget[agent_id] -= self.gift_reward


    def send_gift_replenishable_budget(self, agent_id: AgentID, target_agent_id: AgentID) -> None:
        if self.current_budget[agent_id]-self.gift_reward >= 0:
            self.gifts_in_transit[agent_id] -= self.gift_reward
            self.gifts_in_transit[target_agent_id] += self.gift_reward

            self.stats_gift_send[agent_id] += self.gift_reward
            self.current_budget[agent_id] -= self.gift_reward

    def step_finished(
        self, step: int, next_observations: Optional[dict[AgentID, ObsType]] = None
    ) -> None:
        super(Gifting, self).step_finished(step, next_observations)

        # copy the gifts sent and received to the next step
        for agent_id in self.agent_index_to_id.values():
            self.gift_received[agent_id] = self.gifts_in_transit[agent_id]
            self.gifts_in_transit[agent_id] = 0.0

            self.stats_gift_received[agent_id] += self.gift_received[agent_id]

        self.steps += 1
