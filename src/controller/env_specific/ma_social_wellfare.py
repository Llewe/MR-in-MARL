from sys import float_info
from typing import Optional, Union

from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.config.ctrl_config import MaACConfig, MaMATEConfig
from src.controller.actor_critic import ActorCritic
from src.controller.base_rmp import BaseRMP
from src.controller.mate import Mate
from src.envs.coin_game.coin_game import Action
from src.utils.gym_utils import Space


class MaSocialWellfare(BaseRMP):
    """
    This agent is used to manipulate all other agents to go to the middle of the map if its not there own coin.
    """

    last_social_wellfare: float  # t-2
    current_social_wellfare: float  # t-1
    running_social_wellfare: float  # t
    last_rewards: dict[AgentID, float]  # t-1
    current_rewards: dict[AgentID, float]  # t
    avg_last_reward: float  # t-1

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

        self.last_rewards = {}
        self.current_rewards = {}

        self.set_callback(agent_id="player_0", callback=self._man_agent_0)

    def _man_agent_0(
        self, agent_id: AgentID, last_obs: ObsType, last_act: ActionType, reward: float
    ) -> float:
        self.running_social_wellfare += reward
        self.current_rewards[agent_id] = reward

        if self.last_social_wellfare > self.current_social_wellfare:
            # social wellfare decreased -> lets punish someone

            if self.last_rewards[agent_id] > self.avg_last_reward:
                # He did something bad for the others -> lets punish him
                return -self.ma_amount
            elif self.last_rewards[agent_id] < self.avg_last_reward:
                # He got punished because of the others -> lets help him
                return self.ma_amount
            # Else he was not important for the social wellfare -> do nothing
        return 0.0

    def episode_started(self, episode: int) -> None:
        super().episode_started(episode)
        self.last_social_wellfare = float_info.min
        self.current_social_wellfare = float_info.min
        self.running_social_wellfare = 0.0

        for a in self.agents:
            self.last_rewards[a] = 0.0
            self.current_rewards[a] = 0.0

    def step_finished(
        self, step: int, next_observations: Optional[dict[AgentID, ObsType]] = None
    ) -> None:
        super().step_finished(step, next_observations)
        self.last_social_wellfare = self.current_social_wellfare
        self.current_social_wellfare = self.running_social_wellfare

        for a in self.agents:
            self.last_rewards[a] = self.current_rewards[a]

        self.avg_last_reward = self.current_social_wellfare / len(self.agents)
