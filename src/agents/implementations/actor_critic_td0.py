import torch
from gymnasium.spaces import Space
from pettingzoo.utils.env import ObsDict, ActionDict, AgentID, ObsType, ActionType
from torch.distributions import Categorical
from torch.nn.functional import mse_loss
from torch.optim import Optimizer, Adam

from agents.agents_gym_i import IAgentsGym
from config import actor_critic_config


class ActorCriticTd0(IAgentsGym):
    policy_optimizer: dict[AgentID, Optimizer]
    critic_optimizer: dict[AgentID, Optimizer]

    I: dict[AgentID, float]

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ):
        super().init_agents(action_space, observation_space)
        self.policy_optimizer = {
            agent_id: Adam(
                self.policy_networks[agent_id].parameters(),
                lr=actor_critic_config.ACTOR_LR,
            )
            for agent_id in action_space
        }

        self.critic_optimizer = {
            agent_id: Adam(
                self.policy_networks[agent_id].parameters(),
                lr=actor_critic_config.CRITIC_LR,
            )
            for agent_id in observation_space
        }

        self.I = {agent_id: 1 for agent_id in action_space}

    def init_new_episode(self):
        for agent_id in self.policy_networks:
            self.I[agent_id] = 1

    def act(self, agent_id: AgentID, observation: ObsType) -> (ActionType, float):
        # convert state to float tensor, add 1 dimension, allocate tensor on device
        state = torch.from_numpy(observation).float().unsqueeze(0)

        network = self.policy_networks[agent_id]

        # use network to predict action probabilities
        action_probs = network(state)
        state = state.detach()

        # sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample()

        # return action
        return action.item(), m.log_prob(action)

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
        critic_network = self.critic_networks[agent_id]
        policy_network = self.policy_networks[agent_id]
        critic_opt = self.critic_optimizer[agent_id]
        policy_opt = self.policy_optimizer[agent_id]

        state_tensor = torch.from_numpy(last_observation).float().unsqueeze(0)
        new_state_tensor = torch.from_numpy(curr_observation).float().unsqueeze(0)

        # Berechne den TD-Fehler (Temporal Difference Error)
        td_error = (
            reward
            + actor_critic_config.DISCOUNT_FACTOR * critic_network(new_state_tensor)
            - critic_network(state_tensor)
        )

        # Aktualisiere den Critic (Value Function) mit dem TD-Fehler und dem Optimierer

        critic_loss = mse_loss(critic_network(state_tensor),
                                 reward + actor_critic_config.DISCOUNT_FACTOR * critic_network(
                                     new_state_tensor))
        critic_opt.zero_grad()
        critic_loss.backward(retain_graph=True)
        critic_opt.step()

        # Berechne den Advantage (Vorteil) für den Actor
        advantage = td_error

        # Aktualisiere den Actor (Policy) mit dem Advantage und dem Optimierer
        actor_loss = -log_prob * advantage
        policy_opt.zero_grad()
        actor_loss.backward()
        policy_opt.step()

    def act_parallel(self, observations: ObsDict) -> (ActionDict, dict[AgentID:float]):
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
