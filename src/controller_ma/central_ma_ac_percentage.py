from statistics import mean
from typing import Dict

import numpy as np
import torch
from gymnasium.spaces import Box, Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.config.ctrl_config import ACConfig
from src.controller.actor_critic import ActorCritic
from src.interfaces.ma_controller_i import IMaController


class CentralMaAcPercentageConfig(ACConfig):
    UPPER_BOUND: float = 3.0
    LOWER_BOUND: float = 0.0


class CentralMaAcPercentage(ActorCritic, IMaController):
    agent_name = "ma_percentage_controller"
    nr_agents: int

    agent_id_mapping: dict[AgentID, int]

    upper_bound: float
    lower_bound: float

    # stats for logs
    changed_rewards: list[float]

    def __init__(self):
        super().__init__(CentralMaAcPercentageConfig())
        self.changed_rewards = []

        self.upper_bound = self.config.UPPER_BOUND
        self.lower_bound = self.config.LOWER_BOUND

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        policy_network = self.actor_networks[agent_id]
        return policy_network(obs_tensor).detach().numpy()[0]

    def set_agents(self, agents: list[AgentID], observation_space: Space) -> None:
        self.nr_agents = len(agents)

        self.agent_id_mapping = {i: agent_id for i, agent_id in enumerate(agents)}

        action_space = {
            self.agent_name: Box(
                low=self.lower_bound,
                high=self.upper_bound,
                shape=(self.nr_agents,),
                dtype=np.float32,
            )
        }
        observation_space = {self.agent_name: observation_space}

        self.init_agents(action_space, observation_space)

    def update_rewards(
        self,
        obs: ObsType | dict[AgentID, ObsType],
        rewards: dict[AgentID, float],
        metrics: Dict[str, float] | None = None,
    ) -> dict[AgentID, float]:
        percentage_changes: list[float] = self.act(self.agent_name, obs)

        new_rewards: dict[AgentID, float] = rewards.copy()
        changed_reward: float = 0

        for i, percentage in enumerate(percentage_changes):
            agent_id: AgentID = self.agent_id_mapping[i]

            changed_reward += IMaController.distribute_to_others(
                rewards, new_rewards, agent_id, percentage
            )

        # Reward for manipulator
        social_welfare = sum(rewards.values())

        self.changed_rewards.append(changed_reward / self.nr_agents)

        self.step_agent(
            agent_id=self.agent_name,
            last_observation=obs,
            last_action=percentage_changes,
            reward=social_welfare,
            done=False,
            info={},
        )

        return new_rewards

    def epoch_started(self, epoch: int) -> None:
        super().epoch_started(epoch)
        self.changed_rewards.clear()

    def epoch_finished(self, epoch: int, tag: str) -> None:
        super().episode_finished(epoch, tag)
        if self.writer is not None:
            self.writer.add_scalar(
                tag=f"{self.agent_name}/percentage_reward",
                scalar_value=mean(self.changed_rewards),
                global_step=epoch,
            )
