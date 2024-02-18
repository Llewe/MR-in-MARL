from statistics import mean

from gymnasium.spaces import Discrete, Space
from pettingzoo.utils.env import AgentID, ObsType

from src.config.ma_ac_config import CentralMaFixedPercentageConfig
from src.controller.actor_critic import ActorCritic
from src.enums.metrics_e import MetricsE
from src.interfaces.ma_controller_i import IMaController


class CentralMaFixedPercentage(ActorCritic, IMaController):
    agent_name = "central_ma_fixed_percentage_controller"
    nr_agents: int

    agent_id_mapping: dict[AgentID, int]

    # stats for logs
    changed_rewards: list[float]

    def __init__(self):
        super().__init__(CentralMaFixedPercentageConfig())
        self.changed_rewards = []

    def set_agents(self, agents: list[AgentID], observation_space: Space) -> None:
        self.nr_agents = len(agents)

        self.agent_id_mapping = {i: agent_id for i, agent_id in enumerate(agents)}

        action_space = {self.agent_name: Discrete(self.nr_agents)}
        observation_space = {self.agent_name: observation_space}

        self.init_agents(action_space, observation_space)

    def update_rewards(
        self,
        obs: ObsType | dict[AgentID, ObsType],
        rewards: dict[AgentID, float],
        metrics: dict[MetricsE, float] | None = None,
    ) -> dict[AgentID, float]:
        """
        Only central = ma_agents == None, obs is ObsType
        Parameters
        ----------
        metrics
        obs
        rewards
        ma_agents

        Returns
        -------

        """

        punish_agent: int = self.act(self.agent_name, obs)

        # This is like the environment's
        punish_agent_id: AgentID = self.agent_id_mapping[punish_agent]

        new_rewards: dict[AgentID, float] = rewards.copy()

        changed_reward: float = IMaController.distribute_to_others(
            rewards, new_rewards, punish_agent_id, self.config.PERCENTAGE
        )

        # Reward for manipulator
        social_welfare = sum(rewards.values())

        self.changed_rewards.append(changed_reward)

        self.step_agent(
            agent_id=self.agent_name,
            last_observation=obs,
            last_action=punish_agent,
            reward=social_welfare,
            done=False,
            info={},
        )

        return new_rewards

    def epoch_started(self, epoch: int) -> None:
        super().epoch_started(epoch)
        self.changed_rewards.clear()

    def epoch_finished(self, epoch: int, tag: str) -> None:
        super().episode_finished(epoch, tag)
        if self.writer is not None:
            self.writer.add_scalar(
                tag=f"{self.agent_name}/percentage_reward",
                scalar_value=mean(self.changed_rewards),
                global_step=epoch,
            )
