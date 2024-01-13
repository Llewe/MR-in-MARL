from itertools import islice
from statistics import mean
from typing import Dict, List

import numpy as np
import torch
from gymnasium.spaces import Box, Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.config.ctrl_config import ACConfig
from src.controller.actor_critic import ActorCritic
from src.enums.metrics_e import MetricsE
from src.interfaces.ma_controller_i import IMaController


class IndividualMaACPGlobalMetricConfig(ACConfig):
    UPPER_BOUND: float = 0.5
    LOWER_BOUND: float = -0.5

    METRICS: List[MetricsE] = [MetricsE.EFFICIENCY]

    NR_MA_AGENTS: int = 1


class IndividualMaACPGlobalMetric(ActorCritic, IMaController):
    agent_name = "individual_ma_ac_p_global_metric_controller"

    nr_agents: int

    agent_id_mapping: dict[AgentID, int]

    ma_agents: dict[
        AgentID, str
    ]  # used for internal names so that the logging is to two different graphs

    upper_bound: float
    lower_bound: float

    # stats for logs
    changed_rewards: List[float]

    writer: SummaryWriter

    def __init__(self):
        super().__init__(IndividualMaACPGlobalMetricConfig())
        self.changed_rewards = []

        self.upper_bound = self.config.UPPER_BOUND
        self.lower_bound = self.config.LOWER_BOUND
        self.nr_ma_agents = self.config.NR_MA_AGENTS

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        policy_network = self.actor_networks[agent_id]
        return policy_network(obs_tensor).detach().numpy()[0]

    def set_agents(
        self, agents: List[AgentID], observation_space: dict[AgentID, Space]
    ) -> None:
        self.nr_agents = len(agents)
        self.ma_agents = {a: f"ma_{a}" for a in islice(agents, self.nr_ma_agents)}

        self.agent_id_mapping = {i: agent_id for i, agent_id in enumerate(agents)}

        action_space = {
            ma: Box(
                low=self.lower_bound,
                high=self.upper_bound,
                shape=(self.nr_agents,),
                dtype=np.float32,
            )
            for a, ma in self.ma_agents.items()
        }
        observation_space = {
            ma: observation_space[a] for a, ma in self.ma_agents.items()
        }

        self.init_agents(action_space, observation_space)

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

        percentage_changes: Dict[AgentID, List[float]] = self.act_parallel(filtered_obs)

        new_rewards: Dict[AgentID, float] = {}

        missing_reward: float = 0

        for a, percentages in percentage_changes.items():
            for i, percentage in enumerate(percentages):
                agent_id: AgentID = self.agent_id_mapping[i]
                percentage_reward = rewards[agent_id] * percentage
                new_rewards[agent_id] = rewards[agent_id] - percentage_reward
                missing_reward += percentage_reward

        remove_from_all = missing_reward / self.nr_agents

        for a in new_rewards:
            new_rewards[a] += remove_from_all

        self.changed_rewards.append(remove_from_all)

        self.step_agent_parallel(
            last_observations=filtered_obs,
            last_actions=percentage_changes,
            rewards=metric_rewards,
            dones={ma: False for a, ma in self.ma_agents.items()},
            infos={},
        )

        return new_rewards

    def episode_started(self, episode: int) -> None:
        pass

    def episode_finished(self, episode: int, tag: str) -> None:
        pass

    def set_logger(self, writer: SummaryWriter) -> None:
        self.writer = writer

    def epoch_started(self, epoch: int) -> None:
        self.changed_rewards.clear()

    def epoch_finished(self, epoch: int, tag: str) -> None:
        if self.writer is not None:
            self.writer.add_scalar(
                tag=f"{self.agent_name}/changed_rewards",
                scalar_value=mean(self.changed_rewards),
                global_step=epoch,
            )
