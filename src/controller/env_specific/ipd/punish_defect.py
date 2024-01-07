from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.config.ctrl_config import MaConfig
from src.controller.actor_critic import ActorCritic
from src.controller.base_rmp import BaseRMP


class MaIpdPunishDefect(BaseRMP):
    """
    This agent uses reward manipulation to punish the opponent for defecting.
    """

    ma_amount: float

    # Actions from environment
    COOPERATE: int = 0
    DEFECT: int = 1

    def __init__(self, config: MaConfig):
        super().__init__(ActorCritic, config)
        self.ma_amount = config.MANIPULATION_AMOUNT
        self.set_callback(agent_id="player_0", callback=self._man_agent_0)

    def _man_agent_0(
        self, agent_id: AgentID, last_obs: ObsType, last_act: ActionType, reward: float
    ) -> float:
        # check if we want to manipulate this agent
        if last_act == self.DEFECT:
            # manipulate the reward
            return self.ma_amount
        else:
            return 0.0
