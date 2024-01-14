from statistics import mean
from typing import Dict

from gymnasium.spaces import Discrete, Space
from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.config.ctrl_config import ACConfig
from src.controller.actor_critic import ActorCritic
from src.enums.metrics_e import MetricsE
from src.interfaces.ma_controller_i import IMaController


class CentralMaHeuristicCoinGame(IMaController):
    agent_name = "central_ma_heuristic_coin_game_controller"

    changed_rewards: list[float]
    writer: SummaryWriter

    def __init__(self):
        self.changed_rewards = []

    def update_rewards(
        self,
        obs: ObsType | Dict[AgentID, ObsType],
        rewards: Dict[AgentID, float],
        metrics: Dict[MetricsE, float] | None = None,
    ) -> Dict[AgentID, float]:
        min_reward = min(rewards.values())

        if min_reward < 0:
            # find coin owner
            coin_owner = None

            ma_rewards = {}

            for a, r in rewards.items():
                if r == min_reward:
                    coin_owner = a
                    break
            # give all except coin owner a small punishment

            sw: float = sum(rewards.values())
            other_agents = len(rewards) - 1
            reward_for_each = sw / other_agents

            for a, r in rewards.items():
                if a == coin_owner:
                    ma_rewards[a] = 0.0
                    self.changed_rewards.append(0.0 - r)
                else:
                    ma_rewards[a] = reward_for_each
                    self.changed_rewards.append(r - reward_for_each)

            return ma_rewards

        for r in rewards.values():
            self.changed_rewards.append(0)

        return rewards

    def set_agents(self, agents: list[AgentID], observation_space: Space) -> None:
        pass

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
