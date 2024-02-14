# reference
# https://github.com/thomyphan/emergent-cooperation


from typing import Any, Optional

import numpy
import torch
from gymnasium import Space
from pettingzoo.utils.env import AgentID, ObsType
from torch import Tensor

from src.config.ctrl_config import MateConfig
from src.controller.actor_critic import ActorCritic


class Mate(ActorCritic):
    """
    This implementation is a simplified version of the original implementation.
    It removes parts of the neighborhood since this is not relevant for the coin game
    and ipd.
    """

    mate_mode: MateConfig.Mode
    defect_mode: MateConfig.DefectMode
    token_value: float

    trust_request_matrix: numpy.ndarray
    trust_response_matrix: numpy.ndarray

    last_rewards_observed: dict[AgentID, list[float]]

    request_messages_sent: float
    response_messages_sent: float
    steps_in_epoch: float

    nr_agents: int

    agent_id_mapping: dict[AgentID, int]
    neighborhood: dict[AgentID, list[AgentID]]

    def __init__(self, config: MateConfig):
        super(Mate, self).__init__(config)

        self.mate_mode = config.MODE
        self.defect_mode = config.DEFECT_MODE
        self.token_value = config.TOKEN_VALUE

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        super(Mate, self).init_agents(action_space, observation_space)
        self.nr_agents = len(action_space.keys())

        self.agent_id_mapping = {}
        self.neighborhood = (
            {}
        )  # TODO neighborhood needs to change if i wanna use it in harvest

        for i, agent_id in enumerate(action_space.keys()):
            self.agent_id_mapping[agent_id] = i
            self.neighborhood[agent_id] = [
                x for x in action_space.keys() if x != agent_id
            ]

        self.trust_request_matrix: numpy.ndarray = numpy.zeros(
            (self.nr_agents, self.nr_agents), dtype=numpy.float64
        )
        self.trust_response_matrix: numpy.ndarray = numpy.zeros(
            (self.nr_agents, self.nr_agents), dtype=numpy.float64
        )

        self.last_rewards_observed = {agent_id: [] for agent_id in action_space.keys()}

    def epoch_started(self, epoch: int) -> None:
        super(Mate, self).epoch_started(epoch)

        self.request_messages_sent = 0.0
        self.response_messages_sent = 0.0
        self.steps_in_epoch = 0.0

    def epoch_finished(self, epoch: int, tag: str) -> None:
        super(Mate, self).epoch_finished(epoch, tag)

        if self.writer is not None and self.steps_in_epoch > 0:
            self.writer.add_scalar(
                f"{tag}/mate/requests_sent",
                self.request_messages_sent / self.steps_in_epoch,
                epoch,
            )

            self.writer.add_scalar(
                f"{tag}/mate/responses_sent",
                self.response_messages_sent / self.steps_in_epoch,
                epoch,
            )

    def episode_started(self, episode: int) -> None:
        super(Mate, self).episode_started(episode)
        self.last_rewards_observed = {
            agent_id: [] for agent_id in self.actor_networks.keys()
        }

    def step_finished(
        self, step: int, next_observations: Optional[dict[AgentID, ObsType]] = None
    ) -> None:
        super(Mate, self).step_finished(step, next_observations)
        self.mate(next_observations)
        self.steps_in_epoch += 1

    def get_state_values(
        self,
        last_obs_index: int,
        next_observations: dict[AgentID, Tensor],
    ) -> dict[AgentID, tuple[float, float]]:
        """
        Calculates the state values for the given observations
        Parameters
        ----------
        last_observations: dict[AgentID, Tensor]
        next_observations: dict[AgentID, Tensor]

        Returns
        -------
        dict[AgentID, tuple[float, float]]
            A dictionary containing the state values for each agent
            Float 1: Value of the last observation
            Float 2: Value of the next observation

        """
        state_values: dict[AgentID, tuple[float, float]] = {
            agent_id: (
                self.step_info[agent_id].values[last_obs_index].item(),
                self.critic_networks[agent_id](
                    next_observations[agent_id].detach()
                ).item(),
            )
            for agent_id in self.agent_id_mapping.keys()
        }

        return state_values

    def can_rely_on(
        self,
        agent_id: AgentID,
        reward: float,
        v_old: float,
        v_new: float,
    ):  # history, next_history is missing
        if self.mate_mode == MateConfig.Mode.STATIC_MODE:
            is_empty = self.last_rewards_observed[agent_id]
            if not is_empty:
                # This is a change to the original implementation. The original
                # implementation has a bug here (at least i think so ;)).
                self.last_rewards_observed[agent_id].append(reward)
                return True
            last_reward = numpy.mean(self.last_rewards_observed[agent_id])
            self.last_rewards_observed[agent_id].append(reward)
            return reward >= last_reward
        if self.mate_mode == MateConfig.Mode.TD_ERROR_MODE:
            return reward + self.config.DISCOUNT_FACTOR * v_new - v_old >= 0
        if self.mate_mode == MateConfig.Mode.VALUE_DECOMPOSE_MODE:
            return False

    def get_neighborhood(self, agent_id: AgentID) -> list[AgentID]:
        if (
            self.step_info[agent_id].info[-1]
            and "neighbours" in self.step_info[agent_id].info[-1]
        ):
            return self.step_info[agent_id].info[-1]["neighbours"]
        else:
            return self.neighborhood[agent_id]

    def mate(self, next_observations: Optional[dict[AgentID, ObsType]] = None):
        original_rewards: dict[AgentID, float] = {
            a: self.step_info[a].rewards[-1] for a in self.step_info.keys()
        }
        mate_rewards: dict[AgentID, float] = {
            a: self.step_info[a].rewards[-1] for a in self.step_info.keys()
        }

        self.trust_request_matrix[:] = 0.0
        self.trust_response_matrix[:] = 0.0

        last_obs_index: int

        if next_observations is None:
            last_obs_index = -2
            next_observations = {
                agent_id: self.step_info[agent_id].observations[-1]
                for agent_id in self.step_info.keys()
            }

        else:
            last_obs_index = -1
            next_observations = {  # type: ignore
                agent_id: torch.tensor(
                    next_observations[agent_id], dtype=torch.float64
                ).unsqueeze(0)
                for agent_id in self.step_info.keys()
            }

        state_values: dict[AgentID, tuple[float, float]]
        if self.mate_mode == MateConfig.Mode.TD_ERROR_MODE:
            state_values = self.get_state_values(last_obs_index, next_observations)
        else:
            state_values = {a: (0.0, 0.0) for a in self.step_info.keys()}

        for agent_id, i in self.agent_id_mapping.items():
            reward = original_rewards[agent_id]
            neighborhood = self.get_neighborhood(agent_id)

            if self.can_rely_on(
                agent_id, reward, state_values[agent_id][0], state_values[agent_id][1]
            ):  # Analyze the "winners" of that step
                for neighbor in neighborhood:
                    j = self.agent_id_mapping[neighbor]
                    self.trust_request_matrix[j][i] += self.token_value
                    self.request_messages_sent += 1  # logging

        # 2. Send trust responses
        for agent_id, i in self.agent_id_mapping.items():
            neighborhood = self.get_neighborhood(agent_id)

            trust_requests = [
                self.trust_request_matrix[i][self.agent_id_mapping[j]]
                for j in neighborhood
            ]
            if len(trust_requests) > 0:
                mate_rewards[agent_id] += numpy.max(trust_requests)

            if len(neighborhood) > 0:
                if self.can_rely_on(
                    agent_id,
                    mate_rewards[agent_id],
                    state_values[agent_id][0],
                    state_values[agent_id][1],
                ):
                    accept_trust = self.token_value
                else:
                    accept_trust = -self.token_value
                for neighbor in neighborhood:
                    j = self.agent_id_mapping[neighbor]
                    if self.trust_request_matrix[i][j] > 0:
                        self.trust_response_matrix[j][i] = accept_trust
                        if accept_trust > 0:
                            self.response_messages_sent += 1  # logging
        # 3. Receive trust responses
        for agent_id, i in self.agent_id_mapping.items():
            neighborhood = self.get_neighborhood(agent_id)
            trust_responses = self.trust_response_matrix[i]

            if len(neighborhood) > 0 and trust_responses.any():
                filtered_trust_responses: list[Any] = [
                    trust_responses[self.agent_id_mapping[x]]
                    for x in neighborhood
                    if abs(trust_responses[self.agent_id_mapping[x]]) > 0
                ]
                if len(filtered_trust_responses) > 0:
                    mate_rewards[agent_id] += min(filtered_trust_responses)

        #  write mate rewards to step_info
        for agent_id in self.agent_id_mapping.keys():
            self.step_info[agent_id].rewards[-1] = mate_rewards[agent_id]
