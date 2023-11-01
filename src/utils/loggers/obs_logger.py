from collections import defaultdict

from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter


class ObsLogger:
    summary_writer: SummaryWriter

    log_obs_buffer: dict[AgentID, list[ObsType]] = defaultdict(list)

    def __init__(self, summary_writer: SummaryWriter):
        self.summary_writer = summary_writer

    def add_observation(self, agent_id: AgentID, obs: ObsType):
        self.log_obs_buffer[agent_id].append(obs)

    def clear_buffer(self):
        self.log_obs_buffer.clear()

    def log_episode(self, episode: int, tag: str):
        pass
