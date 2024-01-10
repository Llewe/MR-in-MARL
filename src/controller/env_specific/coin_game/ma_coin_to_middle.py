from typing import Union

from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.config.ctrl_config import MaACConfig, MaMATEConfig
from src.controller.actor_critic import ActorCritic
from src.controller.base_rmp import BaseRMP
from src.controller.mate import Mate
from src.envs.coin_game.coin_game import Action
from src.utils.gym_utils import Space


class MaCoinToMiddle(BaseRMP):
    """
    This agent is used to manipulate all other agents to go to the middle of the map if its not there own coin.
    """

    ma_amount: float

    agent_id_map: dict[AgentID, int]

    middle: list[tuple[int, int]]  # there can be multiple middles if grid size is even

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

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        super().init_agents(action_space, observation_space)
        for i, agent_id in enumerate(self.agents):
            self.agent_id_map[agent_id] = i

        grid_size = observation_space["player_0"].high[0][0]

        if grid_size % 2 == 0:
            self.middle = [
                (grid_size // 2, grid_size // 2),
                (grid_size // 2, grid_size // 2 - 1),
                (grid_size // 2 - 1, grid_size // 2),
                (grid_size // 2 - 1, grid_size // 2 - 1),
            ]
        else:
            self.middle = [(grid_size // 2, grid_size // 2)]

    def _man_agent_0(
        self, agent_id: AgentID, last_obs: ObsType, last_act: ActionType, reward: float
    ) -> float:
        # check if we want to manipulate this agent

        if last_obs[-1] == 1:
            # target agent was the coin owner -> no manipulation
            return 0.0

        target_offset: int = 2 * self.agent_id_map[agent_id]

        target_pos: tuple[int, int] = (
            last_obs[target_offset],
            last_obs[target_offset + 1],
        )

        action = Action(last_act)

        next_pos: tuple[int, int]

        # if action == Action.LEFT:
        #     next_pos = (target_pos[0] - 1, target_pos[1])
        # elif action == Action.RIGHT:
        #     next_pos = (target_pos[0] + 1, target_pos[1])
        # elif action == Action.UP:
        #     next_pos = (target_pos[0], target_pos[1] + 1)
        # elif action == Action.DOWN:
        #     next_pos = (target_pos[0], target_pos[1] - 1)
        # elif action == Action.NONE:
        #     next_pos = target_pos
        # else:
        #     # unknown action -> no manipulation
        #     return 0.0

        if target_pos in self.middle:
            # target agent is already in the middle -> no manipulation
            return 0.0

        # target agent is not in the middle -> manipulate
        return -self.ma_amount
