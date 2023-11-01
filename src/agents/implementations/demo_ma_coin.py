import numpy as np
from pettingzoo.utils.env import AgentID, ObsType, ActionType

from src.agents.implementations.base_agents.mr_agent_a2c2 import MRAgentA2C2
from src.config.ctrl_configs.demo_ma_coin import DemoMaCoinConfig


class DemoMaCoin(MRAgentA2C2):
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
