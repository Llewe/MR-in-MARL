from statistics import mean

from gymnasium.spaces import Discrete, Space
from pettingzoo.utils.env import AgentID, ObsType

from src.config.ctrl_config import ACConfig
from src.controller.actor_critic import ActorCritic
from src.interfaces.ma_controller_i import IMaController


class CentralMaFixedPercentageConfig(ACConfig):
    PERCENTAGE: float = 0.8


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
        self, obs: ObsType | dict[AgentID, ObsType], rewards: dict[AgentID, float]
    ) -> dict[AgentID, float]:
        """
        Only central = ma_agents == None, obs is ObsType
        Parameters
        ----------
        obs
        rewards
        ma_agents

        Returns
        -------

        """

        punish_agent: int = self.act(self.agent_name, obs)

        # This is like the environment's
        punish_agent_id: AgentID = self.agent_id_mapping[punish_agent]

        percentage_reward: float = rewards[punish_agent_id] * self.config.PERCENTAGE
        add_to_others = percentage_reward / (self.nr_agents - 1)

        for agent_id in self.agent_id_mapping.values():
            if agent_id != punish_agent_id:
                rewards[agent_id] += add_to_others
            else:
                rewards[agent_id] -= percentage_reward

        # Reward for manipulator
        social_welfare = sum(rewards.values())

        self.changed_rewards.append(percentage_reward)

        self.step_agent(
            agent_id=self.agent_name,
            last_observation=obs,
            last_action=punish_agent,
            reward=social_welfare,
            done=False,
            info={},
        )

        return rewards

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
