from statistics import mean
from typing import Dict, List

import numpy as np
import torch
from gymnasium.spaces import Box, Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.config.ctrl_config import ACConfig
from src.controller.actor_critic import ActorCritic
from src.controller_ma.utils.ma_ac import MaAc, MaAcConfig
from src.interfaces.ma_controller_i import IMaController


class CentralMaAcPercentage(MaAc):
    agent_name = "ma_percentage_controller"

    central_agent_id: str

    def __init__(self):
        config = MaAcConfig()
        config.NR_MA_AGENTS = 1  # always only one agent because its central
        super().__init__(config)

    def set_agents(
        self, agents: List[AgentID], observation_space: Space | dict[AgentID, Space]
    ) -> None:
        super().set_agents(agents, observation_space)
        self.central_agent_id = list(self.ma_agents.values())[0]

    def update_rewards(
        self,
        obs: ObsType | Dict[AgentID, ObsType],
        rewards: Dict[AgentID, float],
        metrics: Dict[str, float] | None = None,
    ) -> Dict[AgentID, float]:
        filtered_obs = {self.central_agent_id: obs}
        value_to_optimize = {
            self.central_agent_id: sum(rewards.values())
        }  # social wellfare

        new_rewards, percentage_changes = self.proxy_step(filtered_obs, rewards)

        self.step_agent_parallel(
            last_observations=filtered_obs,
            last_actions=percentage_changes,
            rewards=value_to_optimize,
            dones={self.central_agent_id: False},
            infos={},
        )

        return new_rewards
