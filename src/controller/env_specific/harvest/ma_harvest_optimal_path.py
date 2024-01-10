from itertools import product
from typing import Union

from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.config.ctrl_config import MaACConfig, MaMATEConfig
from src.controller.actor_critic import ActorCritic
from src.controller.base_rmp import BaseRMP
from src.controller.mate import Mate
from src.envs.coin_game.coin_game import Action
from src.envs.harvest.harvest import ObsTiles
from src.utils.gym_utils import Space


class MaHarvestOptimalPath(BaseRMP):

    """
    This agent is used to manipulate all other agents to go to the middle of the map if its not there own coin.
    """

    ma_amount: float

    agent_id_map: dict[AgentID, int]

    middle: list[tuple[int, int]]  # there can be multiple middles if grid size is even

    """
         0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4
         1 . . . . A . . . . A . . . . A . . . . A . . . .
         2 . . . A A A . . A A A . . A A A . . A A A . . .
         3 . . . . A . . . . A . . . . A . . . . A . . . .
         4 . . . . . . . . . . . . . . . . . . . . . . . .
         5 . A . . . . A . . . . A . . . . A . . . . A . .
         6 A A A . . A A A . . A A A . . A A A . . A A A .
         7 . A . . . . A . . . . A . . . . A . . . . A . .
         8 . . . . . . . . . . . . . . . . . . . . . . . .
        """

    """
    Index 0: x = -3, y = -3 left top
    Index 1: x = -3, y = -2 left top-1
    Index 2: x = -3, y = -1 left top-2
    Index 3: x = -3, y = 0 left middle
    Index 4: x = -3, y = 1 left bottom-2
    Index 5: x = -3, y = 2 left bottom-1
    Index 6: x = -3, y = 3 left bottom
    Index 7: x = -2, y = -3 left-1 top
    Index 8: x = -2, y = -2 left-1 top-1
    Index 9: x = -2, y = -1 left-1 top-2
    Index 10: x = -2, y = 0 left-1 middle
    Index 11: x = -2, y = 1 left-1 bottom-2
    Index 12: x = -2, y = 2 left-1 bottom-1
    Index 13: x = -2, y = 3 left-1 bottom
    Index 14: x = -1, y = -3 left-2 top
    Index 15: x = -1, y = -2 left-2 top-1
    Index 16: x = -1, y = -1 left-2 top-2
    Index 17: x = -1, y = 0 left-2 middle
    Index 18: x = -1, y = 1 left-2 bottom-2
    Index 19: x = -1, y = 2 left-2 bottom-1
    Index 20: x = -1, y = 3 left-2 bottom
    Index 21: x = 0, y = -3 middle top
    Index 22: x = 0, y = -2 middle top-1
    Index 23: x = 0, y = -1 middle top-2
    Index 24: x = 0, y = 0 middle middle
    Index 25: x = 0, y = 1 middle bottom-2
    Index 26: x = 0, y = 2 middle bottom-1
    Index 27: x = 0, y = 3 middle bottom
    Index 28: x = 1, y = -3 right-2 top
    Index 29: x = 1, y = -2 right-2 top-1
    Index 30: x = 1, y = -1 right-2 top-2
    Index 31: x = 1, y = 0 right-2 middle
    Index 32: x = 1, y = 1 right-2 bottom-2
    Index 33: x = 1, y = 2 right-2 bottom-1
    Index 34: x = 1, y = 3 right-2 bottom
    Index 35: x = 2, y = -3 right-1 top
    Index 36: x = 2, y = -2 right-1 top-1
    Index 37: x = 2, y = -1  right-1 top-2
    Index 38: x = 2, y = 0 right-1 middle
    Index 39: x = 2, y = 1 right-1 bottom-2
    Index 40: x = 2, y = 2 right-1 bottom-1
    Index 41: x = 2, y = 3 right-1 bottom
    Index 42: x = 3, y = -3 right top
    Index 43: x = 3, y = -2 right top-1
    Index 44: x = 3, y = -1 right top-2
    Index 45: x = 3, y = 0 right middle
    Index 46: x = 3, y = 1 right bottom-2
    Index 47: x = 3, y = 2 right bottom-1
    Index 48: x = 3, y = 3 right bottom
    """
    top_row: list[int] = [0, 7, 14, 21, 28, 35, 42]
    top_row_1: list[int] = [1, 8, 15, 22, 29, 36, 43]
    top_row_2: list[int] = [2, 9, 16, 23, 30, 37, 44]
    middle_row: list[int] = [3, 10, 17, 24, 31, 38, 45]
    bottom_row_2: list[int] = [4, 11, 18, 25, 32, 39, 46]
    bottom_row_1: list[int] = [5, 12, 19, 26, 33, 40, 47]
    bottom_row: list[int] = [6, 13, 20, 27, 34, 41, 48]

    farm_routs: dict[int, list[tuple[int, int]]] = {
        0: [(x, 1) for x in range(1, 24)]
        + [(23, y) for y in range(2, 7)]
        + [(x, 7) for x in range(23, 0, -1)]
        + [(1, y) for y in range(6, 1, -1)],
        1: [(x, 3) for x in range(3, 22)]
        + [(21, y) for y in range(4, 6)]
        + [(x, 5) for x in range(21, 2, -1)]
        + [(3, y) for y in range(4, 3, -1)],
    }

    agent_groups_mapping: dict[AgentID, int]

    def __init__(
        self,
        config: Union[MaACConfig, MaMATEConfig],
    ):
        if isinstance(config, MaACConfig):
            super().__init__(ActorCritic, config)
        elif isinstance(config, MaMATEConfig):
            super().__init__(Mate, config)
        else:
            raise ValueError("Unknown config type")
        self.ma_amount = config.MANIPULATION_AMOUNT

        self.set_callback(agent_id="player_0", callback=self._man_agent_0)

        self.agent_id_map = {}
        self.agent_groups_mapping = {}

    def isSeeingTopBorderRows(self, obs: ObsType, border_rows: int) -> bool:
        if border_rows < 1:
            raise ValueError("border_rows must be at least 1")
        for i in self.top_row:
            if obs[i] != ObsTiles.OUTSIDE:
                return False
        if border_rows < 2:
            return True
        for i in self.top_row_1:
            if obs[i] != ObsTiles.OUTSIDE:
                return False
        if border_rows < 3:
            return True
        for i in self.top_row_2:
            if obs[i] != ObsTiles.OUTSIDE:
                return False
        return True

    def isSeeingBottomBorderRows(self, obs: ObsType, border_rows: int) -> bool:
        if border_rows < 1:
            raise ValueError("border_rows must be at least 1")
        for i in self.bottom_row:
            if obs[i] != ObsTiles.OUTSIDE:
                return False
        if border_rows < 2:
            return True
        for i in self.bottom_row_1:
            if obs[i] != ObsTiles.OUTSIDE:
                return False
        if border_rows < 3:
            return True
        for i in self.bottom_row_2:
            if obs[i] != ObsTiles.OUTSIDE:
                return False
        return True

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        super().init_agents(action_space, observation_space)
        for i, agent_id in enumerate(self.agents):
            self.agent_id_map[agent_id] = i

        s = True
        for agent_id in self.agents:
            self.agent_groups_mapping[agent_id] = 0 if s else 1
            s = not s

    def _man_agent_0(
        self, agent_id: AgentID, last_obs: ObsType, last_act: ActionType, reward: float
    ) -> float:
        # check if we want to manipulate this agent

        if self.agent_groups_mapping[agent_id] == 0:
            if self.isSeeingTopBorderRows(last_obs, 2):
                return 0.0
        if self.agent_groups_mapping[agent_id] == 1:
            if self.isSeeingBottomBorderRows(last_obs, 2):
                return 0.0

        # target agent is not in the right part of the map
        return -self.ma_amount
