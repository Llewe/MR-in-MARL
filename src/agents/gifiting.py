from gymnasium import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType

from src.agents.a2c import A2C
from src.config.ctrl_config import GiftingConfig


class Gifting(A2C):
    gift_mode: GiftingConfig.Mode
    gift_reward: float

    def __init__(self, config: GiftingConfig):
        super(Gifting, self).__init__(config)

        self.gift_mode = config.MODE
        self.gift_reward = config.GIFT_REWARD

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        for agent_id, space in action_space.items():
            action_space[agent_id] = Space(
                space.shape, (space.n + 1,), space.min_n, space.max_n
            )

        super(Gifting, self).init_agents(action_space, observation_space)

    def epoch_started(self, epoch: int) -> None:
        super(Gifting, self).epoch_started(epoch)

    def epoch_finished(self, epoch: int) -> None:
        super(Gifting, self).epoch_finished(epoch)

    def step_agent(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        super(Gifting, self).step_agent(
            agent_id, last_observation, last_action, reward, done
        )

    def step_finished(self, step: int) -> None:
        super(Gifting, self).step_finished(step)
