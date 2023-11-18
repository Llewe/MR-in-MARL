from collections import defaultdict
from dataclasses import dataclass, field
from typing import List

import numpy as np
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.env import AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.utils.loggers.obs_logger import ObsLogger


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
    agent_vel_pos: VelPos
    other_entity_vel_pos: dict[str | AgentID, VelPos] = field(default_factory=dict)


class SimpleEnvLogger(ObsLogger):
    env: SimpleEnv

    log_vel_pos_info_buffer: dict[AgentID, list[PosInfo]] = defaultdict(list)

    entity_groups: dict[str | AgentID, str]
    first_entity: str | AgentID

    def __init__(self, env: SimpleEnv, summary_writer: SummaryWriter):
        super().__init__(summary_writer)

        self.env = env

        self.entity_groups = SimpleEnvLogger._get_agent_group_mapping(
            [entity.name for entity in self.env.world.entities]
        )

        self.first_entity = self.env.world.entities[0].name

    def add_observation(self, agent_id: AgentID, obs: ObsType):
        super().add_observation(agent_id, obs)

        pos_info: PosInfo = PosInfo()

        for entity in self.env.world.entities:
            vel_pos: VelPos = VelPos(
                x=entity.state.p_pos[0],
                y=entity.state.p_pos[1],
                vel_x=entity.state.p_vel[0],
                vel_y=entity.state.p_vel[1],
            )

            if entity.name == agent_id:
                pos_info.agent_vel_pos = vel_pos
            else:
                pos_info.other_entity_vel_pos[entity.name] = vel_pos

        self.log_vel_pos_info_buffer[agent_id].append(pos_info)

    def clear_buffer(self):
        super().clear_buffer()

        self.log_vel_pos_info_buffer.clear()

    def log_episode(self, episode: int, tag: str):
        super().log_episode(episode, tag)

        mean_distances: dict[
            str | AgentID, dict[str | AgentID, float]
        ] = self._calculate_square_distances()

        self._log_mean_individual_distances(
            episode=episode, tag=tag, distances=mean_distances
        )
        self._log_mean_group_distances(
            episode=episode, tag=tag, distances=mean_distances[self.first_entity]
        )

    def _log_mean_group_distances(
        self, episode: int, tag: str, distances: dict[str | AgentID, float]
    ):
        # its not a "cluster" distance
        group_distances: dict[str, list[float]] = defaultdict(list)

        # only log the distances of first agent to all other agents
        # (since in theory they should be the same)

        for agent_id, mean_square_distance in distances.items():
            group_distances[agent_id].append(mean_square_distance)

        for group_id, d in group_distances.items():
            mean_distance = np.mean(d)
            self.summary_writer.add_scalar(
                tag=f"mpe-{tag}/msd-group/{group_id}-to-{self.first_entity}",
                scalar_value=mean_distance,
                global_step=episode,
            )

    @staticmethod
    def _get_agent_group_mapping(
        entity_names: List[str | AgentID],
    ) -> dict[str | AgentID, str]:
        agent_types: dict[str | AgentID, str] = {}
        for agent_id in entity_names:
            group_id: str
            if agent_id.startswith("agent"):
                group_id = "agent"
            elif agent_id.startswith("adversary"):
                group_id = "adversary"
            elif agent_id.startswith("landmark"):
                group_id = "landmark"
            else:
                raise ValueError(
                    f"Unknown agent type: {agent_id}. "
                    f"Please just add the new group to the if-else chain."
                )
            agent_types[agent_id] = group_id

        return agent_types

    def _log_mean_individual_distances(
        self,
        episode: int,
        tag: str,
        distances: dict[str | AgentID, dict[str | AgentID, float]],
    ):
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

        for agent_id in self.log_vel_pos_info_buffer.keys():
            pos_info_list: list[PosInfo] = self.log_vel_pos_info_buffer[agent_id]

            pos_agent_list: list[np.array] = []
            pos_entity_list: dict[str | AgentID, list[np.array]] = defaultdict(list)
            for info in pos_info_list:
                pos_agent_list.append([info.agent_vel_pos.x, info.agent_vel_pos.y])
                for k, v in info.other_entity_vel_pos.items():
                    pos_entity_list[k].append([v.x, v.y])

            mean_distances[agent_id] = {
                k: self._mean_square_distance(v, pos_entity_list[agent_id])
                for k, v in pos_entity_list.items()
                if k != agent_id
            }

        return mean_distances

    @staticmethod
    def _mean_square_distance(pos1: list[np.array], pos2: list[np.array]) -> float:
        # Convert lists to NumPy arrays
        pos1_np = np.array(pos1)
        pos2_np = np.array(pos2)

        # Calculate squared distances using NumPy
        distances = np.sum((pos1_np[:, np.newaxis] - pos2_np) ** 2, axis=2)

        # Calculate mean distance
        mean_distance = np.mean(distances)

        return mean_distance
