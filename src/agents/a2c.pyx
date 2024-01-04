import logging
from dataclasses import dataclass
from os import makedirs
from os.path import join
from typing import Optional

import numpy as np
import torch
from gymnasium import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from src.agents.utils.network import ActorNetwork, CriticNetwork
from src.agents.utils.reward_normalization import RewardNormalization
from src.config.ctrl_config import A2cConfig
from src.interfaces.agents_i import IAgents
from src.utils.gym_utils import get_space_size


class A2C(IAgents):

    @dataclass
    class RolloutBuffer:
        rewards: list[float]
        observations: list[Tensor]
        values: list[Tensor]
        actions: list[ActionType]
        action_probs: list[np.ndarray]

        def __init__(self):
            self.rewards = []
            self.observations = []
            self.values = []
            self.actions = []
            self.action_probs = []

        def add(
            self, reward: float, observation: Tensor, action: ActionType, value: Tensor,
                action_prob: np.ndarray
        ):
            self.rewards.append(reward)
            self.observations.append(observation)
            self.values.append(value)
            self.actions.append(action)
            self.action_probs.append(action_prob)

        def clear(self):
            self.rewards.clear()
            self.observations.clear()
            self.values.clear()
            self.actions.clear()
            self.action_probs.clear()

    config: A2cConfig

    actor_networks: dict[AgentID, ActorNetwork]
    critic_networks: dict[AgentID, CriticNetwork]

    reward_norm: dict[AgentID, RewardNormalization]

    writer: SummaryWriter | None = None

    actor_losses: dict[AgentID, list]
    critic_losses: dict[AgentID, list]

    step_info: dict[AgentID, RolloutBuffer]

    epsilon_initial: float
    epsilon: float
    epsilon_final: float
    epsilon_decay_rate: float

    def __init__(self, config: A2cConfig):
        self.config = config

        self.epsilon_initial = config.EPSILON_INIT
        self.epsilon = config.EPSILON_INIT
        self.epsilon_final = config.EPSILON_MIN
        self.epsilon_decay_rate = config.EPSILON_DECAY

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ) -> None:
        self.actor_networks = {
            agent_id: ActorNetwork(
                get_space_size(observation_space[agent_id]),
                get_space_size(action_space[agent_id]),
                self.config.ACTOR_HIDDEN_UNITS,
                self.config.ACTOR_LR,
            )
            for agent_id in action_space
        }

        self.critic_networks = {
            agent_id: CriticNetwork(
                get_space_size(observation_space[agent_id]),
                # observation_space[agent_id].shape[0],
                self.config.CRITIC_HIDDEN_UNITS,
                self.config.CRITIC_LR,
            )
            for agent_id in observation_space
        }
        self.step_info = {agent_id: A2C.RolloutBuffer() for agent_id in action_space}

        self.actor_losses = {agent_id: [] for agent_id in self.actor_networks}
        self.critic_losses = {agent_id: [] for agent_id in self.actor_networks}

        self.reward_norm = {
            agent_id: RewardNormalization() for agent_id in self.actor_networks
        }

    def episode_started(self, episode: int) -> None:
        pass

    def episode_finished(self, episode: int, tag: str) -> None:
        pass

    def epoch_started(self, epoch: int) -> None:
        # update epsilon
        self.epsilon = max(
            self.epsilon_final,
            self.epsilon_initial - self.epsilon_decay_rate * epoch,
        )
        # clear buffers
        for agent_id in self.step_info:
            self.step_info[agent_id].clear()
            self.actor_losses[agent_id].clear()
            self.critic_losses[agent_id].clear()

    def epoch_finished(self, epoch: int, tag: str) -> None:
        # training
        for agent_id in self.actor_networks:
            self.learn(agent_id, self.config.DISCOUNT_FACTOR)

        if self.writer is None:
            return

        # logging
        self.writer.add_scalar(
            f"actor_critic/epsilon",
            self.epsilon,
            global_step=epoch,
        )

        for agent_id in self.actor_networks:
            if len(self.actor_losses[agent_id]) > 0:
                self.writer.add_scalar(
                    f"actor_loss/{agent_id}",
                    np.mean(self.actor_losses[agent_id]),
                    global_step=epoch,
                )
                self.actor_losses[agent_id].clear()

            if len(self.critic_losses[agent_id]) > 0:
                self.writer.add_scalar(
                    f"critic_loss/{agent_id}",
                    np.mean(self.critic_losses[agent_id]),
                    global_step=epoch,
                )
                self.critic_losses[agent_id].clear()

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        if explore and self.epsilon > 0 and np.random.rand() < self.epsilon:
            action = np.random.choice(self.actor_networks[agent_id].num_actions)
            # log_prob = -np.log(1.0 / self.actor_networks[agent_id].action_space.n)
        else:
            policy_network = self.actor_networks[agent_id]

            # convert observation to float tensor, add 1 dimension, allocate tensor on device

            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            action_probs = policy_network(obs_tensor)  # [0].detach().numpy()
            m = Categorical(probs=action_probs)
            action_t: Tensor = m.sample()

            # log_prob = m.log_prob(action)
            action = action_t.item()

        return action

    def local_probs(self, agent_id: AgentID, obs_tensor: Tensor) -> np.ndarray:
        action_probs = self.actor_networks[agent_id](obs_tensor.detach())

        return action_probs.detach().numpy()[0]

    def step_agent(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        if self.config.REWARD_NORMALIZATION:
            scaled_reward: float = self.reward_norm[agent_id].normalize(reward)
        else:
            scaled_reward = reward
        obs_curr: Tensor = torch.tensor(last_observation, dtype=torch.float32).unsqueeze(0)
        local_probs = self.local_probs(agent_id, obs_curr)
        self.step_info[agent_id].add(
            reward=scaled_reward,
            observation=obs_curr,
            action=last_action,
            value=self.critic_networks[agent_id](obs_curr.detach()),
            action_prob=local_probs,
        )

    def step_finished(
        self, step: int, next_observations: Optional[dict[AgentID, ObsType]] = None
    ) -> None:
        pass

    @staticmethod
    def compute_returns(rewards, gamma: float):
        discounted_returns = np.zeros_like(rewards, dtype=np.float32)
        running_add: float = 0.0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_returns[t] = running_add

        # TODO not sure why i added this maybe it helped with MPE envs but it is bad for MATE
        # # Adding a small constant to avoid division by zero
        # mean_returns = np.mean(discounted_returns)
        # std_returns = np.std(discounted_returns) + 1e-8
        #
        # # Normalize using mean and standard deviation with added constant
        # discounted_returns -= mean_returns
        # discounted_returns /= std_returns

        return discounted_returns

    def _update_critic(self, agent_id, gamma: float, returns) -> None:
        critic = self.critic_networks[agent_id]

        critic_values = torch.stack(self.step_info[agent_id].values).squeeze()

        critic_loss = mse_loss(returns.detach(), critic_values)

        self.critic_losses[agent_id].append(critic_loss.detach())

        critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.config.CLIP_NORM)
        critic.optimizer.step()

    def _update_actor(self, agent_id, gamma: float, returns) -> None:
        actor = self.actor_networks[agent_id]
        critic = self.critic_networks[agent_id]

        obs = torch.stack(self.step_info[agent_id].observations)

        actions = torch.tensor(self.step_info[agent_id].actions, dtype=torch.int64)

        critic_values = torch.stack(self.step_info[agent_id].values).squeeze().detach()

        actor_probs = actor(obs.detach())

        advantages = returns.detach() - critic_values

        m1 = Categorical(actor_probs)
        actor_loss = (-m1.log_prob(actions) * advantages).sum()

        self.actor_losses[agent_id].append(actor_loss.detach())

        actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), self.config.CLIP_NORM)
        actor.optimizer.step()

    def learn(self, agent_id: AgentID, gamma: float) -> None:
        if len(self.step_info[agent_id].rewards) == 0:
            self.step_info[agent_id].clear()
            print("No rewards")
            return

        returns = torch.tensor(
            self.compute_returns(self.step_info[agent_id].rewards, gamma),
            dtype=torch.float32,
        )
        self._update_critic(agent_id, gamma, returns)
        self._update_actor(agent_id, gamma, returns)

        self.step_info[agent_id].clear()

    def save(self, path: str) -> None:
        logging.info(f"Saving actor critic to {path}")

        makedirs(path)
        for ids in self.actor_networks:
            actor_net = self.actor_networks[ids]

            model_path = join(
                path,
                f"actor_net_{ids}.pth",
            )
            torch.save(actor_net.state_dict(), model_path)

        for ids in self.actor_networks:
            critic_net = self.critic_networks[ids]
            model_path = join(
                path,
                f"critic_net_{ids}.pth",
            )

            torch.save(critic_net.state_dict(), model_path)

    def load(self, path: str) -> None:
        logging.info(f"Loading actor critic from {path}")

        for ids in self.actor_networks:
            model_path = join(
                path,
                f"actor_net_{ids}.pth",
            )
            self.actor_networks[ids].load_state_dict(torch.load(model_path))
            self.actor_networks[ids].eval()

        for ids in self.actor_networks:
            model_path = join(
                path,
                f"critic_net_{ids}.pth",
            )
            self.critic_networks[ids].load_state_dict(torch.load(model_path))
            self.critic_networks[ids].eval()

    def set_logger(self, writer: SummaryWriter) -> None:
        self.writer = writer
