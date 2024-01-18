from pettingzoo.utils.env import AgentID
from torch.utils.tensorboard import SummaryWriter

from src.enums import EnvType


def log_efficiency(
    epoch: int,
    tag: str,
    summary_writer: SummaryWriter,
    rewards: dict[AgentID, list[float]],
    nr_episodes: int,
) -> None:
    """
    Log efficiency
    Parameters
    ----------
    epoch: int
    tag: str
    summary_writer: SummaryWriter
    rewards: dict[AgentID, list[float]]
        rewards of the agents. Use the real rewards, not the normalized nor mate ones.
    nr_episodes: int
    Returns
    -------

    """

    u: float = 0
    for agent_id, reward in rewards.items():
        u += sum(reward)
    u /= nr_episodes
    summary_writer.add_scalar(f"{tag}/efficiency", u, global_step=epoch)
