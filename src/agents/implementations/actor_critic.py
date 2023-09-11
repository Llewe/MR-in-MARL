import logging
from os import makedirs
from os.path import join

import numpy as np
import torch
from gymnasium import Space
from pettingzoo.utils.env import ObsDict, ActionDict, AgentID, ObsType, ActionType
from torch.distributions import Categorical
from torch.nn import Module, Linear, Sequential, ELU
from torch.nn.functional import mse_loss
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter

from src.agents.agents_i import IAgents
from src.agents.implementations.utils.reward_normalization import RewardNormalization
from src.config import actor_critic_config
from torch import Tensor

# https://medium.com/geekculture/actor-critic-implementing-actor-critic-methods-82efb998c273


def preprocessing_module(nr_input_features, nr_hidden_units):
    return Sequential(
        Linear(nr_input_features, nr_hidden_units),
        ELU(),
        Linear(nr_hidden_units, nr_hidden_units),
        ELU(),
    )


class ActorNetwork(Module):
    # Takes in observations and outputs actions
    def __init__(
        self,
        observation_space,
        action_space: Space,
        hidden_units: int,
        learning_rate: float,
    ):
        super(ActorNetwork, self).__init__()
        self.action_space = action_space
        self.fc_net = preprocessing_module(observation_space, hidden_units)
        self.action_head = Linear(hidden_units, action_space.n)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    # forward pass
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return softmax(self.action_head(x), dim=-1)


class CriticNetwork(Module):
    # Takes in state
    def __init__(self, observation_space, hidden_units: int, learning_rate: float):
        super(CriticNetwork, self).__init__()

        self.fc_net = preprocessing_module(observation_space, hidden_units)
        self.value_head = Linear(hidden_units, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_net(x)
        return self.value_head(x)


class ActorCritic(IAgents):
    actor_networks: dict[AgentID, ActorNetwork]
    critic_networks: dict[AgentID, CriticNetwork]

    reward_norm: dict[AgentID, RewardNormalization]

    returns: dict[AgentID, float]
    writer: SummaryWriter | None = None
    current_episode: int

    actor_losses = {}
    critic_losses = {}

    epsilon_initial = 1.0
    epsilon = epsilon_initial
    epsilon_final = 0.08  # Set to a small value close to zero
    epsilon_decay_rate = 0.001

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ):
        self.actor_networks = {
            agent_id: ActorNetwork(
                observation_space[agent_id].shape[0],
                action_space[agent_id],
                actor_critic_config.ACTOR_HIDDEN_UNITS,
                actor_critic_config.ACTOR_LR,
            )
            for agent_id in action_space
        }

        self.critic_networks = {
            agent_id: CriticNetwork(
                observation_space[agent_id].shape[0],
                actor_critic_config.CRITIC_HIDDEN_UNITS,
                actor_critic_config.CRITIC_LR,
            )
            for agent_id in observation_space
        }
        self.current_episode = 0

        self.actor_losses = {agent_id: [] for agent_id in self.actor_networks}
        self.critic_losses = {agent_id: [] for agent_id in self.actor_networks}

        self.reward_norm = {
            agent_id: RewardNormalization() for agent_id in self.actor_networks
        }

    def init_new_episode(self):
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

        self.current_episode += 1

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        if explore and np.random.rand() < self.epsilon:
            action = np.random.choice(self.actor_networks[agent_id].action_space.n)
            # log_prob = -np.log(1.0 / self.actor_networks[agent_id].action_space.n)
        else:
            policy_network = self.actor_networks[agent_id]

            # convert observation to float tensor, add 1 dimension, allocate tensor on device
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)

            action_probs = policy_network(obs_tensor)  # [0].detach().numpy()
            m = Categorical(probs=action_probs)
            action = m.sample()

            # log_prob = m.log_prob(action)
            action = action.item()

        if self.writer:
            self.writer.add_histogram(
                f"actions/{agent_id}",
                action,
                global_step=self.current_episode,
                max_bins=10,
            )

        return action

    def _update_actor(
        self,
        agent_id: AgentID,
        actor_net: ActorNetwork,
        last_action: ActionType,
        obs_curr: Tensor,
        advantage: Tensor,
    ):
        action_probs = actor_net(obs_curr)

        m1 = Categorical(action_probs)
        actor_loss = -m1.log_prob(torch.tensor(last_action)) * advantage.detach()

        # only for logging
        self.actor_losses[agent_id].append(actor_loss.detach().numpy())

        actor_net.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            actor_net.parameters(), actor_critic_config.CLIP_NORM
        )
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

        value_target: Tensor = scaled_reward + gamma * value_curr * (1 - done)

        loss = mse_loss(
            value_target, value_last
        )  # TODO here im not sure if if it needs to be value_last or value_curr

        # only for logging
        self.critic_losses[agent_id].append(loss.detach().numpy())

        critic_net.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            critic_net.parameters(), actor_critic_config.CLIP_NORM
        )
        critic_net.optimizer.step()

        advantage = value_target - value_last

        return advantage.detach()

    def update(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        curr_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
        gamma: float,
    ) -> None:
        scaled_reward: float = self.reward_norm[agent_id].normalize(reward)

        actor_net: ActorNetwork = self.actor_networks[agent_id]
        critic_net: CriticNetwork = self.critic_networks[agent_id]

        obs_last: Tensor = torch.from_numpy(last_observation).float().unsqueeze(0)
        obs_curr: Tensor = torch.from_numpy(curr_observation).float().unsqueeze(0)

        advantage: Tensor = self._update_critic(
            agent_id=agent_id,
            critic_net=critic_net,
            obs_last=obs_last,
            obs_curr=obs_curr,
            scaled_reward=scaled_reward,
            done=done,
            gamma=gamma,
        )

        self._update_actor(
            agent_id=agent_id,
            actor_net=actor_net,
            last_action=last_action,
            obs_curr=obs_curr,
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
