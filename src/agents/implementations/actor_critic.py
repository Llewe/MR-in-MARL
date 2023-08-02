import torch
from gymnasium import Space
from pettingzoo.utils.env import ObsDict, ActionDict, AgentID, ObsType, ActionType
from torch.distributions import Categorical
from torch.nn import Module, Linear, Sequential, ELU, ReLU
from torch.nn.functional import mse_loss
from torch.nn.functional import softmax
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from agents.agents_i import IAgents
from config import actor_critic_config


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
    returns: dict[AgentID, float]
    writer: SummaryWriter
    current_episode: int

    actor_losses = {}
    critic_losses = {}

    epsilon_initial = 1.0
    epsilon = epsilon_initial
    epsilon_final = 0.1  # Set to a small value close to zero
    epsilon_decay_rate = 0.0001

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

    def act(self, agent_id: AgentID, observation: ObsType) -> (ActionType, float):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actor_networks[agent_id].action_space.n)
            log_prob = -np.log(1.0 / self.actor_networks[agent_id].action_space.n)
        else:
            policy_network = self.actor_networks[agent_id]

            # convert observation to float tensor, add 1 dimension, allocate tensor on device
            state = torch.from_numpy(observation).float().unsqueeze(0)
            # state.requires_grad = True

            # use network to predict action probabilities
            # state = state.detach()

            action_probs = policy_network(state)  # [0].detach().numpy()
            m = Categorical(probs=action_probs)
            action = m.sample()

            # action_probs = policy_network(state)[0].detach().numpy()
            # # print("action_probs.shape: ", action_probs.shape)
            # # print("policy_network.action_space: ", policy_network.action_space)
            # action = numpy.random.choice(policy_network.action_space.n, p=action_probs)

            # sample an action using the probability distribution

            # print("action: ", action)
            # print("c_action: ", c_action)

            # return action
            log_prob = m.log_prob(action)
            action = action.item()

        self.writer.add_histogram(
            f"actions/{agent_id}", action, global_step=self.current_episode, max_bins=10
        )

        return action, log_prob

    mean_reward = 0.0
    std_reward = 1.0

    def update(
        self,
        agent_id: AgentID,
        last_observation: ObsType,
        curr_observation: ObsType,
        last_action: ActionType,
        reward: float,
        done: bool,
        log_prob: float,
        gamma: float,
    ) -> None:
        # Reward normalization
        self.mean_reward = (self.mean_reward * self.current_episode + reward) / (
            self.current_episode + 1
        )
        self.std_reward = np.sqrt(
            (
                (self.std_reward**2) * self.current_episode
                + (reward - self.mean_reward) ** 2
            )
            / (self.current_episode + 1)
        )

        # Normalize the reward
        normalized_reward = (reward - self.mean_reward) / self.std_reward

        reward = normalized_reward

        actor_net: ActorNetwork = self.actor_networks[agent_id]
        critic_net: CriticNetwork = self.critic_networks[agent_id]

        value_last_state = torch.from_numpy(last_observation).float().unsqueeze(0)
        value_curr_state = torch.from_numpy(curr_observation).float().unsqueeze(0)

        value_current_state = critic_net(value_last_state).squeeze()
        value_next_state = critic_net(value_curr_state).squeeze()

        target_value = reward + gamma * value_next_state * (1 - done)

        # update critic

        critic_loss = mse_loss(target_value, value_current_state)

        self.critic_losses[agent_id].append(critic_loss.detach().numpy())

        critic_net.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            critic_net.parameters(), actor_critic_config.CLIP_NORM
        )
        critic_net.optimizer.step()

        # update actor
        action_probs = actor_net(value_curr_state)

        advantage = target_value - value_current_state

        m1 = Categorical(action_probs)
        actor_loss = -m1.log_prob(torch.tensor(last_action)) * advantage.detach()

        self.actor_losses[agent_id].append(actor_loss.detach().numpy())

        actor_net.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            actor_net.parameters(), actor_critic_config.CLIP_NORM
        )
        actor_net.optimizer.step()

    def act_parallel(self, observations: ObsDict) -> ActionDict:
        raise NotImplementedError

    def update_parallel(
        self,
        observations: ObsDict,
        next_observations: ObsDict,
        actions: ActionDict,
        rewards: dict[AgentID:float],
        dones: dict[AgentID:bool],
        log_probs: dict[AgentID:float],
        gamma: float,
    ) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError

    def set_logger(self, writer: SummaryWriter) -> None:
        self.writer = writer
