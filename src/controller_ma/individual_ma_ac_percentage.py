from typing import Dict

from pettingzoo.utils.env import AgentID, ObsType

from src.config.ma_ac_config import MaAcConfig
from src.controller_ma.utils.ma_ac import MaAc
from src.enums.metrics_e import MetricsE


class IndividualMaAcPercentage(MaAc):
    agent_name = "individual_ma_percentage_controller"

    def __init__(self):
        super().__init__(MaAcConfig())

    def update_rewards(
        self,
        obs: ObsType | Dict[AgentID, ObsType],
        rewards: Dict[AgentID, float],
        metrics: Dict[MetricsE, float] | None = None,
    ) -> Dict[AgentID, float]:
        filtered_obs = {ma: obs[a] for a, ma in self.ma_agents.items()}
        value_to_optimize = {ma: rewards[a] for a, ma in self.ma_agents.items()}

        new_rewards, percentage_changes = self.proxy_step(filtered_obs, rewards)

        self.step_agent_parallel(
            last_observations=filtered_obs,
            last_actions=percentage_changes,
            rewards=value_to_optimize,
            dones={ma: False for a, ma in self.ma_agents.items()},
            infos={},
        )

        return new_rewards
