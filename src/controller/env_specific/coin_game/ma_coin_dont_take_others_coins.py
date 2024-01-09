from typing import Optional, Union

from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.config.ctrl_config import MaACConfig, MaMATEConfig
from src.controller.actor_critic import ActorCritic
from src.controller.base_rmp import BaseRMP
from src.controller.mate import Mate
from src.envs.coin_game.coin_game import Action
from src.utils.gym_utils import Space


class MaCoinDontTakeOthersCoins(BaseRMP):
    """
    This agent is used to manipulate all other agents to go to the middle of the map if its not there own coin.
    """

    last_rewards: dict[AgentID, float]  # t-1
    someone_missing_a_coin: bool

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

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        super().init_agents(action_space, observation_space)

    def _man_agent_0(
        self, agent_id: AgentID, last_obs: ObsType, last_act: ActionType, reward: float
    ) -> float:
        last_reward = self.last_rewards[agent_id]
        self.last_rewards[agent_id] = reward
        # check if we want to manipulate this agent
        if self.someone_missing_a_coin:
            if last_reward > 0:
                # He did something bad for the others -> lets punish him
                return -self.ma_amount
            elif last_reward < 0:
                #### THIS SHIFTES some of the punishment to the other agents
                # He got punished because of the others -> lets help him
                return self.ma_amount

        else:
            return 0.0

    def episode_started(self, episode: int) -> None:
        super().episode_started(episode)
        self.someone_missing_a_coin = False

    def step_finished(
        self, step: int, next_observations: Optional[dict[AgentID, ObsType]] = None
    ) -> None:
        self.someone_missing_a_coin = False
        for r in self.last_rewards:
            if r < 0:
                self.someone_missing_a_coin = True
                break
