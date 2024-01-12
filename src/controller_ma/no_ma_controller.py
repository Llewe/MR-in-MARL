from gymnasium import Space
from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.interfaces.ma_controller_i import IMaController


class NoMaController(IMaController):
    def episode_started(self, episode: int) -> None:
        pass

    def episode_finished(self, episode: int, tag: str) -> None:
        pass

    def epoch_started(self, epoch: int) -> None:
        pass

    def epoch_finished(self, epoch: int, tag: str) -> None:
        pass

    def update_rewards(
        self,
        obs: ObsType | dict[AgentID, ObsType],
        rewards: dict[AgentID, float],
    ) -> dict[AgentID, float]:
        return rewards

    def set_agents(
        self, agents: list[AgentID], observation_space: dict[AgentID, Space] | Space
    ) -> None:
        pass

    def set_logger(self, writer: SummaryWriter) -> None:
        pass
