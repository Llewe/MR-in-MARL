from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.agents.mr_agent_a2c import MRAgentA2C
from src.config.ctrl_config import DemoMaCoinConfig


class DemoMaCoin(MRAgentA2C):
    def __init__(self, config: DemoMaCoinConfig):
        super().__init__(config)

        self.set_callback(agent_id="player_0", callback=self._man_agent_0)

    def _man_agent_0(
        self, agent_id: AgentID, last_obs: ObsType, last_act: ActionType, reward: float
    ) -> float:
        # check if we wanna manipulate this agent

        manipulation = 0.0
        if reward > 0.0:
            reward_p0 = last_obs[6]
            reward_p1 = last_obs[13]
            reward_p2 = last_obs[20]
            reward_p3 = last_obs[27]

            if reward_p0 < 0.0:
                manipulation = -self.config.MANIPULATION_AMOUNT * reward

        return manipulation
