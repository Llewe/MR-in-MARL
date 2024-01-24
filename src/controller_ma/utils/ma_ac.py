import statistics
from abc import ABC
from enum import Enum
from itertools import islice
from statistics import mean
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box, Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch import Tensor, clamp
from torch.distributions import Beta
from torch.utils.tensorboard import SummaryWriter

from src.config.ctrl_config import ACConfig
from src.controller.actor_critic import ActorCritic
from src.interfaces.ma_controller_i import IMaController


class MaAcConfig(ACConfig):
    UPPER_BOUND: float = 2.0
    LOWER_BOUND: float = 0.0

    NR_MA_AGENTS: int = 1

    BETA_PROB_CONCENTRATION: float = 2.0

    class DISTRIBUTION(Enum):
        DIRECT = "direct"
        BETA = "beta"
        GAUSSIAN = "gaussian"
        FMAX = "fmax"

    PERCENTAGE_MODE: bool = True

    USE_BETA_DISTRIBUTION: bool = False

    UPDATE_EVERY_X_EPISODES: int = 5

    DIST_TYPE: DISTRIBUTION = DISTRIBUTION.FMAX


class MaAc(ActorCritic, IMaController, ABC):
    agent_name = "ma_ac"

    nr_agents: int

    agent_id_mapping: dict[AgentID, int]

    ma_agents: dict[
        AgentID, str
    ]  # used for internal names so that the logging is to two different graphs

    upper_bound: float
    lower_bound: float

    # stats for logs
    changed_rewards: List[float]

    update_every_x_episodes: int
    dist_type: MaAcConfig.DISTRIBUTION

    writer: SummaryWriter

    def __init__(self, config: MaAcConfig):
        # config.EPSILON_INIT = 0.0
        # config.EPSILON_MIN = 0.00
        # config.EPSILON_DECAY = 9.75e-05
        # config.DISCOUNT_FACTOR = 0.95

        super().__init__(config)
        self.changed_rewards = []

        self.upper_bound = self.config.UPPER_BOUND
        self.lower_bound = self.config.LOWER_BOUND
        self.nr_ma_agents = self.config.NR_MA_AGENTS
        self.update_every_x_episodes = self.config.UPDATE_EVERY_X_EPISODES
        self.dist_type = self.config.DIST_TYPE

        torch.set_default_dtype(torch.float64)

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        if explore and self.epsilon > 0 and np.random.rand() < self.epsilon:
            actions = np.random.rand(self.actor_networks[agent_id].num_actions).tolist()
            # log_prob = -np.log(1.0 / self.actor_networks[agent_id].action_space.n)
            return actions
        else:
            obs_tensor = torch.tensor(observation, dtype=torch.float64).unsqueeze(0)

            policy_network = self.actor_networks[agent_id]

            action_probabilities = policy_network(obs_tensor).detach()[0]

            if np.isnan(action_probabilities).any():
                print(f"obs: {observation}")
                print(f"obs_tensor: {obs_tensor}")
                print(f"action_probabilities: {action_probabilities}")
                raise ValueError("action_probabilities is nan")
            action_probabilities = torch.clamp(
                action_probabilities,
                min=1e-8,
                max=1 - 1e-8,  # help to avoid nan from log(0)
            )

            if self.dist_type == MaAcConfig.DISTRIBUTION.BETA:
                alpha, beta = self.get_alpha_beta(action_probabilities)

                c = Beta(alpha, beta).sample()
                return c.tolist()
            else:
                return action_probabilities.tolist()

    def get_alpha_beta(self, action_probabilities: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        concentration
        action_probabilities

        Returns
        -------

        """

        # Calculate shape parameters
        e = 1e-8
        action_probabilities = torch.clamp(action_probabilities, min=e, max=1.0 - e)

        alpha = action_probabilities * self.config.BETA_PROB_CONCENTRATION
        beta = (1 - action_probabilities) * self.config.BETA_PROB_CONCENTRATION
        #
        alpha = clamp(alpha, min=e, max=self.config.BETA_PROB_CONCENTRATION)
        beta = clamp(beta, min=e, max=self.config.BETA_PROB_CONCENTRATION)

        if torch.isnan(alpha).any() or torch.isnan(beta).any():
            print(f"alpha: {alpha}")
            print(f"beta: {beta}")
            print(f"action_probabilities: {action_probabilities}")
            raise ValueError("alpha or beta is nan")

        return alpha, beta

    def update_actor(self, agent_id, gamma: float, returns) -> None:
        actor = self.actor_networks[agent_id]

        obs = torch.stack(self.step_info[agent_id].observations)

        actions = torch.tensor(self.step_info[agent_id].actions, dtype=torch.float32)

        critic_values = torch.stack(self.step_info[agent_id].values).squeeze().detach()

        actor_probs = actor(obs.detach())
        advantages = returns.detach() - critic_values

        if self.dist_type == MaAcConfig.DISTRIBUTION.BETA:
            alpha, beta = self.get_alpha_beta(actions)
            m1 = Beta(alpha, beta)
            log_prob = torch.sum(m1.log_prob(actor_probs), dim=-1)
            actor_loss = (-log_prob * advantages).sum()
        elif self.dist_type == MaAcConfig.DISTRIBUTION.GAUSSIAN:
            log_probs = torch.sum(
                -0.5 * ((actions - actor_probs) ** 2), dim=-1
            )  # Assuming Gaussian policy
            actor_loss = (-log_probs * advantages).sum()
        elif self.dist_type == MaAcConfig.DISTRIBUTION.FMAX:
            log_probs = torch.sum(F.log_softmax(actor_probs, dim=1), dim=-1)

            actor_loss = (-log_probs * advantages).sum()
        elif self.dist_type == MaAcConfig.DISTRIBUTION.DIRECT:
            actor_probs = torch.clamp(actor_probs, min=1e-8, max=1 - 1e-8)
            log_probs = torch.sum(actor_probs * torch.log(actor_probs), dim=-1)
            actor_loss = (-log_probs * advantages).sum()
        else:
            raise ValueError("Unknown distribution type")

        actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), self.config.CLIP_NORM)
        actor.optimizer.step()

    def set_agents(
        self, agents: List[AgentID], observation_space: Space | dict[AgentID, Space]
    ) -> None:
        self.nr_agents = len(agents)
        self.ma_agents = {a: f"ma_{a}" for a in islice(agents, self.nr_ma_agents)}

        self.agent_id_mapping = {i: agent_id for i, agent_id in enumerate(agents)}

        action_space = {
            ma: Box(
                low=self.lower_bound,
                high=self.upper_bound,
                shape=(self.nr_agents,),
                dtype=np.float32,
            )
            for a, ma in self.ma_agents.items()
        }
        if isinstance(observation_space, Space):
            observation_space = {
                ma: observation_space for a, ma in self.ma_agents.items()
            }
        else:
            observation_space = {
                ma: observation_space[a] for a, ma in self.ma_agents.items()
            }

        self.init_agents(action_space, observation_space)

    def proxy_step(
        self, obs: Dict[AgentID, ObsType], rewards: Dict[AgentID, float]
    ) -> Tuple[Dict[AgentID, float], Dict[AgentID, List[float]]]:
        # if central there will be just one agent otherwise there can be multiple
        percentage_changes: Dict[AgentID, List[float]] = self.act_parallel(obs)

        # scale percentage to desired bounds
        scaled_percentage_changes = {
            a: [
                self.lower_bound + ((self.upper_bound - self.lower_bound) * f)
                for f in p
            ]
            for a, p in percentage_changes.items()
        }
        # build average of all percentage changes
        transposed_lists = zip(*scaled_percentage_changes.values())
        average_percentage_changes = list(map(statistics.mean, transposed_lists))

        new_rewards: dict[AgentID, float] = rewards.copy()

        changed_reward: float = 0  # for logging

        for i, percentage in enumerate(average_percentage_changes):
            agent_id: AgentID = self.agent_id_mapping[i]

            changed_reward += IMaController.distribute_to_others(
                rewards, new_rewards, agent_id, percentage
            )
        self.changed_rewards.append(changed_reward)

        return new_rewards, percentage_changes

    def set_logger(self, writer: SummaryWriter) -> None:
        self.writer = writer

    def epoch_started(self, epoch: int) -> None:
        super().epoch_started(epoch)
        self.changed_rewards.clear()

    def epoch_finished(self, epoch: int, tag: str) -> None:
        super().epoch_finished(epoch, tag)
        if self.writer is not None:
            self.writer.add_scalar(
                tag=f"{self.agent_name}/changed_rewards",
                scalar_value=mean(self.changed_rewards),
                global_step=epoch,
            )

    def episode_started(self, episode: int) -> None:
        super().episode_started(episode)
        e = episode - 1
        if self.update_every_x_episodes > 0 and e % self.update_every_x_episodes == 0:
            for agent_id in self.actor_networks:
                self.learn(agent_id, self.config.DISCOUNT_FACTOR)

            for agent_id in self.step_info:
                self.step_info[agent_id].clear()
                self.actor_losses[agent_id].clear()
                self.critic_losses[agent_id].clear()
