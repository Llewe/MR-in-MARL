import gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__doc__ = """
original source: https://github.com/thomyphan/emergent-cooperation/

"""

from gymnasium import Space
from pettingzoo.utils.env import AgentID, ObsDict, ObsType, ActionType, ActionDict
from torch.distributions import MultivariateNormal
from config import actor_critic_config

device = torch.device(actor_critic_config.TORCH_DEVICE)


def assertEquals(first, second):
    assert first == second, "Expected {}, got {}".format(first, second)


def preprocessing_module(nr_input_features, nr_hidden_units, last_layer_units):
    return nn.Sequential(
        nn.Linear(nr_input_features, nr_hidden_units, device=device),
        nn.ELU(),
        nn.Linear(nr_hidden_units, nr_hidden_units, device=device),
        nn.ELU(),
        nn.Linear(nr_hidden_units, last_layer_units, device=device),
    )


class ActorNet(nn.Module):
    def __init__(self, input_dim, nr_actions, nr_hidden_units, learning_rate):
        super(ActorNet, self).__init__()
        self.nr_input_features = input_dim
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, nr_hidden_units),
            nn.ELU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ELU(),
            nn.Linear(nr_hidden_units, nr_actions),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # return F.softmax(self.fc_net(x), dim=-1)
        return self.fc_net(x)


class CriticNet(nn.Module):
    def __init__(self, input_dim, nr_hidden_units, learning_rate):
        super(CriticNet, self).__init__()
        self.nr_input_features = input_dim
        self.nr_hidden_units = nr_hidden_units
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, nr_hidden_units),
            nn.ELU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ELU(),
            nn.Linear(nr_hidden_units, 1),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        return self.fc_net(x)


def calculate_nr_input_features(input_space):
    if isinstance(input_space, gymnasium.spaces.box.Box):
        print(input_space.shape)
        return int(np.prod(input_space.shape))
    elif isinstance(input_space, gymnasium.spaces.Discrete):
        return input_space.n
    else:
        print(input_space)
        raise ValueError(f"Unsupported input space type {type(input_space)}")


class ActorCritic:
    def __init__(
        self,
        agent_names: list[AgentID],  # specific for petting zoo
        observation_dims: list[Space],
        action_dims: list[Space],
        actor_hidden_units: int,
        critic_hidden_units: int,
        actor_lr: float,
        critic_lr: float,
        continuous_actions=False,
    ):
        self.agent_names = agent_names

        self.actors = {}
        self.critics = {}

        self.action_boxes = {}

        # Initialize actor and critic networks for each agent
        for agent_name, observation_dim, action_dim in zip(
            agent_names, observation_dims, action_dims
        ):
            observation_features = calculate_nr_input_features(observation_dim)
            action_features = calculate_nr_input_features(action_dim)

            self.action_boxes[agent_name] = action_dim

            print(
                f"shape of {agent_name} = obs {observation_features}, "
                f"act = {action_features}"
            )
            self.actors[agent_name] = ActorNet(
                observation_features, action_features, actor_hidden_units, actor_lr
            )
            self.critics[agent_name] = CriticNet(
                observation_features, critic_hidden_units, critic_lr
            )

    def get_model(self, agent_name: AgentID) -> nn.Module:
        return self.actors[agent_name]

    def act_single(
        self, agent_name: AgentID, observation: ObsType
    ) -> (ActionType, float):
        observation_tensor = torch.tensor(np.array([observation]), dtype=torch.float32)

        policy = self.actors[agent_name](observation_tensor)

        action_box = self.action_boxes[agent_name]

        action_type = action_box.dtype

        if action_type == np.int64:
            policy = F.softmax(policy, dim=-1)
            dist = torch.distributions.Categorical(policy)
            action = dist.sample()
            return_action = action.item()
        elif action_type == np.float32:
            action_mean = torch.squeeze(policy, dim=-1)
            action_var = torch.full(action_box.shape, 0.1 * 0.1)
            cov_mat = torch.diag(action_var)

            dist = MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            # Rescale the input tensor to fit within the Box space
            action = torch.clamp(
                action, torch.tensor(action_box.low), torch.tensor(action_box.high)
            )
            return_action = action.numpy().reshape(action_box.shape)

        else:
            raise ValueError(f"Unsupported action type: {action_type}")

        # print(f"p {policy}, d {dist}, a {action}")
        return return_action, dist.log_prob(action)

    def act(self, observations: ObsDict) -> (ActionDict, dict[AgentID:float]):
        actions = {}
        log_probs = {}
        for agent_name in observations:
            action, log_prob = self.act_single(agent_name, observations[agent_name])

            actions[agent_name] = action
            log_probs[agent_name] = log_prob

        return actions, log_probs

    def evaluate(
        self, observations: ObsDict, actions: ActionDict
    ) -> (dict[AgentID:float], dict[AgentID:float]):
        values = {}
        action_log_probs = {}
        for agent_name in observations:
            obs = observations[agent_name]

            observation_tensor = torch.tensor(obs, dtype=torch.float)
            action_tensor = torch.tensor(actions[agent_name], dtype=torch.long)
            policy = self.actors[agent_name](observation_tensor)
            value = self.critics[agent_name](observation_tensor)

            dist = torch.distributions.Categorical(policy)
            action_log_probs[agent_name] = dist.log_prob(action_tensor)
            values[agent_name] = value.squeeze()

        return action_log_probs, values

    def update_single(
        self,
        agent_name: AgentID,
        observation: ObsType,
        action: ActionType,
        reward: float,
        next_observation: ObsType,
        done: bool,
        log_prob: float,
        gamma,
    ) -> None:
        actor = self.actors[agent_name]
        critic = self.critics[agent_name]

        observation = torch.tensor(np.array([observation]), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_observation = torch.tensor(np.array([next_observation]), dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        # Compute TD targets (critic target)
        with torch.no_grad():
            next_values = critic(next_observation)
            td_targets = reward + gamma * next_values * (1 - done)

        # Compute advantages (critic loss)
        values = critic(observation)
        advantages = td_targets.detach() - values.detach()

        # Compute actor and critic losses
        actor_loss = -(log_prob * advantages).mean()
        critic_loss = F.mse_loss(values, td_targets.detach())

        # Update actor and critic networks
        actor.optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1)
        actor.optimizer.step()

        critic.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1)
        critic.optimizer.step()

    def update(
        self,
        observations: ObsDict,
        actions: ActionDict,
        rewards: dict[AgentID, float],
        next_observations: ObsDict,
        dones: dict[AgentID, bool],
        log_probs,
        gamma,
    ):
        for agent_name in observations:
            self.update_single(
                agent_name,
                observations[agent_name],
                actions[agent_name],
                rewards[agent_name],
                next_observations[agent_name],
                dones[agent_name],
                log_probs[agent_name],
                gamma,
            )

    def preprocess_observation(self, observation: ObsType):
        return torch.tensor(observation, dtype=torch.float)

    def get_value(self, agent_name: AgentID, observation: ObsType):
        observation_tensor = torch.tensor(observation, dtype=torch.float)
        value = self.critics[agent_name](observation_tensor)
        return value.item()
