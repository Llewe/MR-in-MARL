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
from src.config.ctrl_configs.actor_critic_config import ActorCriticConfig
from src.interfaces.agents_i import IAgents
from src.utils.gym_utils import get_space_size


# https://medium.com/geekculture/actor-critic-implementing-actor-critic-methods-82efb998c273


class ActorCriticV2(IAgents):
    config: ActorCriticConfig

    actor_networks: dict[AgentID, ActorNetwork]
    critic_networks: dict[AgentID, CriticNetwork]

    reward_norm: dict[AgentID, RewardNormalization]

    returns: dict[AgentID, float]
    writer: SummaryWriter | None = None
    current_episode: int

    actor_losses = {}
    critic_losses = {}

    epsilon_initial: float
    epsilon: float
    epsilon_final: float
    epsilon_decay_rate: float

    def __init__(self, config: ActorCriticConfig):
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
        self.current_episode = 0

        self.actor_losses = {agent_id: [] for agent_id in self.actor_networks}
        self.critic_losses = {agent_id: [] for agent_id in self.actor_networks}

        self.reward_norm = {
            agent_id: RewardNormalization() for agent_id in self.actor_networks
        }

    def init_new_epoch(self):
        for agent_id in self.actor_networks:
            if len(self.actor_losses[agent_id]) > 0:
                self.writer.add_scalar(
                    f"actor_loss/{agent_id}",
                    np.mean(self.actor_losses[agent_id]),
                    global_step=self.current_episode,
                )

            if len(self.critic_losses[agent_id]) > 0:
                self.writer.add_scalar(
                    f"critic_loss/{agent_id}",
                    np.mean(self.critic_losses[agent_id]),
                    global_step=self.current_episode,
                )

        self.returns = {agent_id: 0.0 for agent_id in self.actor_networks}
        self.actor_losses = {agent_id: [] for agent_id in self.actor_networks}
        self.critic_losses = {agent_id: [] for agent_id in self.actor_networks}

        self.epsilon = max(
            self.epsilon_final,
            self.epsilon_initial - self.epsilon_decay_rate * self.current_episode,
        )
        self.writer.add_scalar(
            f"actor_critic/epsilon",
            self.epsilon,
            global_step=self.current_episode,
        )
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

        # if self.writer:
        #     self.writer.add_histogram(
        #         f"actions/{agent_id}",
        #         action,
        #         global_step=self.current_episode,
        #         max_bins=10,
        #     )

        return action

    def _update_actor(
        self,
        agent_id: AgentID,
        actor_net: ActorNetwork,
        last_action: ActionType,
        obs_last: Tensor,
        advantage: Tensor,
    ):
        action_probs = actor_net(obs_last)
        value = self.critic_networks[agent_id](obs_last).squeeze().detach()

        m1 = Categorical(action_probs)

        adv = torch.tensor(last_action).detach() - value.detach()

        actor_loss = (m1.log_prob(torch.tensor(last_action)) * adv.detach()) ** 2

        # only for logging
        self.actor_losses[agent_id].append(actor_loss.detach().numpy())

        actor_net.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), self.config.CLIP_NORM)
        actor_net.optimizer.step()

    def _update_critic(
        self,
        agent_id: AgentID,
        critic_net: CriticNetwork,
        obs_last: Tensor,
        obs_curr: Tensor,
        scaled_reward: float,
        done: bool,
        gamma: float,
    ):
        value_last: Tensor = critic_net(obs_last).squeeze()
        value_curr: Tensor = critic_net(obs_curr).squeeze()
        value_target: Tensor = torch.tensor(
            scaled_reward, dtype=torch.float
        ) + gamma * value_last * (1 - done)

        loss = mse_loss(
            value_target.detach(), value_last
        )  # TODO here im not sure if if it needs to be value_last or value_curr

        # only for logging
        self.critic_losses[agent_id].append(loss.detach().numpy())

        critic_net.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic_net.parameters(), self.config.CLIP_NORM)
        critic_net.optimizer.step()

        advantage = scaled_reward - value_last.detach()

        return advantage.detach()

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

        actor_net: ActorNetwork = self.actor_networks[agent_id]
        critic_net: CriticNetwork = self.critic_networks[agent_id]

        obs_last: Tensor = torch.from_numpy(last_observation).float().unsqueeze(0)

        advantage: Tensor = self._update_critic(
            agent_id=agent_id,
            critic_net=critic_net,
            obs_last=obs_last,
            scaled_reward=scaled_reward,
            done=done,
            gamma=self.config.DISCOUNT_FACTOR,
        )

        self._update_actor(
            agent_id=agent_id,
            actor_net=actor_net,
            last_action=last_action,
            obs_last=obs_last,
            advantage=advantage,
        )

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
