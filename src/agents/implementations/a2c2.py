import logging
from os import makedirs
from os.path import join

import numpy as np
import torch
from gymnasium import Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter

from src.agents.implementations.utils.network import ActorNetwork, CriticNetwork
from src.agents.implementations.utils.reward_normalization import RewardNormalization
from src.config.ctrl_configs.a2c_config import A2cConfig
from src.config.ctrl_configs.actor_critic_config import ActorCriticConfig
from src.interfaces.agents_i import IAgents
from src.utils.gym_utils import get_space_size


# https://medium.com/geekculture/actor-critic-implementing-actor-critic-methods-82efb998c273


class A2C2(IAgents):
    class RolloutBuffer:
        rewards: list[float]
        observations: list[ObsType]
        actions: list[ActionType]

        def __init__(self):
            self.rewards = []
            self.observations = []
            self.actions = []

        def add(self, reward: float, observation: ObsType, action: ActionType):
            self.rewards.append(reward)
            self.observations.append(observation)
            self.actions.append(action)

        def clear(self):
            self.rewards.clear()
            self.observations.clear()
            self.actions.clear()

    config: A2cConfig
    actor_critic_config = ActorCriticConfig()

    actor_networks: dict[AgentID, ActorNetwork]
    critic_networks: dict[AgentID, CriticNetwork]

    reward_norm: dict[AgentID, RewardNormalization]

    writer: SummaryWriter | None = None
    current_episode: int

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
    ):
        self.actor_networks = {
            agent_id: ActorNetwork(
                get_space_size(observation_space[agent_id]),
                get_space_size(action_space[agent_id]),
                self.actor_critic_config.ACTOR_HIDDEN_UNITS,
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
        self.step_info = {agent_id: A2C2.RolloutBuffer() for agent_id in action_space}

        self.current_episode = 0

        self.actor_losses = {agent_id: [] for agent_id in self.actor_networks}
        self.critic_losses = {agent_id: [] for agent_id in self.actor_networks}

        self.reward_norm = {
            agent_id: RewardNormalization() for agent_id in self.actor_networks
        }

    def init_new_episode(self):
        # update epsilon
        self.epsilon = max(
            self.epsilon_final,
            self.epsilon_initial - self.epsilon_decay_rate * self.current_episode,
        )

        # training
        if (
            self.current_episode > 0
            and self.current_episode % self.config.UPDATE_FREQ == 0
        ):
            for agent_id in self.actor_networks:
                self.learn(agent_id, self.config.DISCOUNT_FACTOR)

        # logging
        self.writer.add_scalar(
            f"actor_critic/epsilon",
            self.epsilon,
            global_step=self.current_episode,
        )

        for agent_id in self.actor_networks:
            if len(self.actor_losses[agent_id]) > 0:
                self.writer.add_scalar(
                    f"actor_loss/{agent_id}",
                    np.mean(self.actor_losses[agent_id]),
                    global_step=self.current_episode,
                )
                self.actor_losses[agent_id].clear()

            if len(self.critic_losses[agent_id]) > 0:
                self.writer.add_scalar(
                    f"critic_loss/{agent_id}",
                    np.mean(self.critic_losses[agent_id]),
                    global_step=self.current_episode,
                )
                self.critic_losses[agent_id].clear()

        self.current_episode += 1

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        if explore and np.random.rand() < self.epsilon:
            action = np.random.choice(self.actor_networks[agent_id].num_actions)
            # log_prob = -np.log(1.0 / self.actor_networks[agent_id].action_space.n)
        else:
            policy_network = self.actor_networks[agent_id]

            # convert observation to float tensor, add 1 dimension, allocate tensor on device
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)

            action_probs = policy_network(obs_tensor).detach()  # [0].detach().numpy()
            m = Categorical(probs=action_probs)
            action = m.sample()

            # log_prob = m.log_prob(action)
            action = action.item()

        return action

    def update(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        curr_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
    ) -> None:
        if self.config.REWARD_NORMALIZATION:
            scaled_reward: float = self.reward_norm[agent_id].normalize(reward)
        else:
            scaled_reward = reward
        obs_curr: Tensor = torch.from_numpy(last_observation).float().unsqueeze(0)

        self.step_info[agent_id].add(scaled_reward, obs_curr, last_action)

    def compute_returns(self, rewards, gamma: float):
        discounted_returns = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_returns[t] = running_add
        # Adding a small constant to avoid division by zero
        mean_returns = np.mean(discounted_returns)
        std_returns = np.std(discounted_returns) + 1e-8

        # Normalize using mean and standard deviation with added constant
        discounted_returns -= mean_returns
        discounted_returns /= std_returns

        return discounted_returns

    def _update_critic(self, agent_id, gamma: float, returns) -> None:
        critic = self.critic_networks[agent_id]

        obs = torch.tensor(
            np.vstack(self.step_info[agent_id].observations), dtype=torch.float32
        ).detach()

        critic_values = critic(obs).squeeze()

        critic_loss = mse_loss(returns.detach(), critic_values)

        self.critic_losses[agent_id].append(critic_loss.detach())

        critic.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), self.config.CLIP_NORM)
        critic.optimizer.step()

    def _update_actor(self, agent_id, gamma: float, returns) -> None:
        actor = self.actor_networks[agent_id]
        critic = self.critic_networks[agent_id]

        obs = torch.tensor(
            np.vstack(self.step_info[agent_id].observations), dtype=torch.float32
        ).detach()

        actions = torch.tensor(
            self.step_info[agent_id].actions, dtype=torch.int64
        ).detach()

        critic_values = critic(obs).squeeze().detach()

        actor_probs = actor(obs)

        advantages = returns.detach() - critic_values.detach()

        m1 = Categorical(actor_probs)
        actor_loss = (-m1.log_prob(actions) * (advantages.detach())).sum()

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
            torch.save(actor_net, model_path)

        for ids in self.actor_networks:
            critic_net = self.critic_networks[ids]
            model_path = join(
                path,
                f"critic_net_{ids}.pth",
            )

            torch.save(critic_net, model_path)

    def load(self, path: str) -> None:
        logging.info(f"Loading actor critic from {path}")

        for ids in self.actor_networks:
            model_path = join(
                path,
                f"actor_net_{ids}.pth",
            )
            self.actor_networks[ids] = torch.load(model_path)

        for ids in self.actor_networks:
            model_path = join(
                path,
                f"critic_net_{ids}.pth",
            )
            self.critic_networks[ids] = torch.load(model_path)

    def set_logger(self, writer: SummaryWriter) -> None:
        self.writer = writer
