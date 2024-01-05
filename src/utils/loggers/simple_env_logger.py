from dataclasses import dataclass, field

import numpy as np
import scipy
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.utils.loggers.obs_logger import IObsLogger


@dataclass
class VelPos:
    x: float
    y: float
    vel_x: float
    vel_y: float

    def __init__(self, x: float, y: float, vel_x: float, vel_y: float):
        self.x = x
        self.y = y
        self.vel_x = vel_x
        self.vel_y = vel_y


@dataclass
class PosInfo:
    vel_pos: dict[str | AgentID, list[VelPos]] = field(default_factory=dict)
    pos: dict[str | AgentID, list[np.array]] = field(default_factory=dict)


class SimpleEnvLogger(IObsLogger):
    env: SimpleEnv

    pos_info: PosInfo

    first_entity: str | AgentID
    last_logged_step: int

    def __init__(self, env: SimpleEnv, summary_writer: SummaryWriter):
        super().__init__(summary_writer)

        self.env = env

        self.last_logged_step = -1

        self.first_entity = self.env.world.entities[0].name

        self.pos_info = PosInfo()

    def add_observation(self, agent_id: AgentID, obs: ObsType):
        if self.env.steps != self.last_logged_step:
            # This  ensures that we only log the data once per step
            self.last_logged_step = self.env.steps

            for entity in self.env.world.entities:
                # vel_pos: VelPos = VelPos(
                #     x=entity.state.p_pos[0],  # x
                #     y=entity.state.p_pos[1],  # y
                #     vel_x=entity.state.p_vel[0],  # vel_x
                #     vel_y=entity.state.p_vel[1],  # vel_y
                # )
                if entity.name not in self.pos_info.pos:
                    self.pos_info.pos[entity.name] = []

                self.pos_info.pos[entity.name].append(
                    np.array([entity.state.p_pos[0], entity.state.p_pos[1]])
                )

    def clear_buffer(self) -> None:
        self.pos_info.vel_pos.clear()
        self.pos_info.pos.clear()
        self.last_logged_step: int = -1

    def log_epoch(self, episode: int, tag: str) -> None:
        mean_distances: dict[
            str | AgentID, dict[str | AgentID, float]
        ] = self._calculate_square_distances()

        self._log_mean_individual_distances(
            episode=episode, tag=tag, distances=mean_distances
        )

    def _log_mean_individual_distances(
        self,
        episode: int,
        tag: str,
        distances: dict[str | AgentID, dict[str | AgentID, float]],
    ) -> None:
        for k, v in distances.items():
            for agent_id, mean_square_distance in v.items():
                self.summary_writer.add_scalar(
                    tag=f"mpe-{tag}/msd/{k}-to-{agent_id}",
                    scalar_value=mean_square_distance,
                    global_step=episode,
                )

    def _calculate_square_distances(
        self,
    ) -> dict[str | AgentID, dict[str | AgentID, float]]:
        """
        Calculate mean squared distance between agent and other entities.
        This function assumes that there is always the same amount of entities in the
        environment.
        Returns
        -------

        """
        mean_distances: dict[str | AgentID, dict[str | AgentID, float]] = {}

        distances_from_agent: list[str | AgentID] = ["agent_0"]

        for agent_id in distances_from_agent:
            mean_distances[agent_id] = {
                k: self._mean_square_distance(self.pos_info.pos[agent_id], v)
                for k, v in self.pos_info.pos.items()
                if k != agent_id
            }
        return mean_distances

    @staticmethod
    def _mean_square_distance(pos1: list[np.array], pos2: list[np.array]) -> float:
        dist = scipy.spatial.distance.cdist(pos1, pos2, metric="sqeuclidean")

        # # Convert lists to NumPy arrays
        # pos1_np = np.array(pos1)
        # pos2_np = np.array(pos2)
        #
        # # Calculate squared distances using NumPy
        # distances = np.sum((pos1_np[:, np.newaxis] - pos2_np) ** 2, axis=2)

        # Calculate mean distance
        mean_distance = np.mean(dist)

        return mean_distance
