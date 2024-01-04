from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.agents.mr_agent_a2c import MRAgentA2C
from src.utils.gym_utils import Space

from src.config.ctrl_config import MaConfig
from src.envs.aec.my_coin_game import Action


class MaIpdPunishDefect(MRAgentA2C):
    """
    This agent uses reward manipulation to punish the opponent for defecting.
    """


    ma_amount: float


    # Actions from environment
    COOPERATE:int = 0
    DEFECT:int = 1

    def __init__(self, config: MaConfig):
        super().__init__(config)
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
        # check if we want to manipulate this agent
        if last_act == self.DEFECT:
            # manipulate the reward
            return -self.ma_amount
        else:
            return 0.0