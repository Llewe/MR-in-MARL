from typing import Dict

from pettingzoo.utils.env import AgentID, ObsType

from src.config.ma_ac_config import IndividualMaACPGlobalMetricConfig
from src.controller_ma.utils.ma_ac import MaAc
from src.enums.metrics_e import MetricsE


class IndividualMaACPGlobalMetric(MaAc):
    agent_name = "individual_ma_ac_p_global_metric_controller"

    def __init__(self):
        super().__init__(IndividualMaACPGlobalMetricConfig())

    def update_rewards(
        self,
        obs: ObsType | Dict[AgentID, ObsType],
        rewards: Dict[AgentID, float],
        metrics: Dict[MetricsE, float] | None,
    ) -> Dict[AgentID, float]:
        assert metrics is not None
        assert len(metrics) == len(self.ma_agents)
        filtered_obs = {ma: obs[a] for a, ma in self.ma_agents.items()}
        metric_rewards = {
            ma: metric
            for (metric_name, metric), (a, ma) in zip(
                metrics.items(), self.ma_agents.items()
            )
        }
        new_rewards, percentage_changes = self.proxy_step(filtered_obs, rewards)

        self.step_agent_parallel(
            last_observations=filtered_obs,
            last_actions=percentage_changes,
            rewards=metric_rewards,
            dones={ma: False for a, ma in self.ma_agents.items()},
            infos={},
        )

        return new_rewards
