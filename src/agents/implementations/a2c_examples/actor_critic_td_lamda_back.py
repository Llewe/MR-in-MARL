import torch
from gymnasium import Space
from pettingzoo.utils.env import ObsDict, ActionDict, AgentID, ObsType, ActionType
from torch.distributions import Categorical
from src.agents.agents_i import IAgents
from src.agents.implementations.utils.policy_network import PolicyNetwork
from src.agents.implementations.utils.state_value_network import StateValueNetwork

ACTOR_LAMBDA = 0.8
CRITIC_LAMBDA = 0.8
DISCOUNT_FACTOR = 0.80
ACTOR_LR = 0.001
CRITIC_LR = 0.001
# https://medium.com/geekculture/actor-critic-implementing-actor-critic-methods-82efb998c273


class ActorCriticTdLamdaBack(IAgents):
    policy_networks: dict[AgentID, PolicyNetwork]
    stateval_networks: dict[AgentID, StateValueNetwork]

    actor_trace: dict[AgentID, list] = {}
    critic_trace: dict[AgentID, list] = {}
    I: dict[AgentID, float] = {}

    def init_agents(
        self,
        action_space: dict[AgentID, Space],
        observation_space: dict[AgentID, Space],
    ):
        self.policy_networks = {
            agent_id: PolicyNetwork(
                observation_space[agent_id].shape[0], action_space[agent_id].n
            )
            for agent_id in action_space
        }

        self.stateval_networks = {
            agent_id: StateValueNetwork(observation_space[agent_id].shape[0])
            for agent_id in observation_space
        }

    def init_new_episode(self):
        for agent_id in self.policy_networks:
            self.actor_trace[agent_id] = []
            self.critic_trace[agent_id] = []
            self.I[agent_id] = 1

    def act(self, agent_id: AgentID, observation: ObsType, explore=True) -> ActionType:
        policy_network = self.policy_networks[agent_id]

        # convert state to float tensor, add 1 dimension, allocate tensor on device
        state = torch.from_numpy(observation).float().unsqueeze(0)
        state.requires_grad = True

        # use network to predict action probabilities
        state = state.detach()
        action_probs = policy_network(state)

        # sample an action using the probability distribution
        m = Categorical(action_probs)
        action = m.sample()

        # return action
        return action.item(), m.log_prob(action)

    def update(self, agent_id: AgentID, last_observation: ObsType,
               curr_observation: ObsType, last_action: ActionType, reward: float,
               done: bool, gamma: float) -> None:
        policy_network = self.policy_networks[agent_id]
        stateval_network = self.stateval_networks[agent_id]

        actor_trace = self.actor_trace[agent_id]
        critic_trace = self.critic_trace[agent_id]

        i = self.I[agent_id]

        # update actor trace
        policy_network.zero_grad()
        log_prob.backward(retain_graph=True)

        if not actor_trace:
            with torch.no_grad():
                for p in policy_network.parameters():
                    # initialize trace
                    trace = i * p.grad
                    actor_trace.append(trace)
        else:
            with torch.no_grad():
                for i, p in enumerate(policy_network.parameters()):
                    # decay trace and shift trace in direction of most recent gradient
                    actor_trace[i] = (
                        actor_trace[i] * ACTOR_LAMBDA * DISCOUNT_FACTOR + i * p.grad
                    )

        # get state value of current state
        state_tensor = torch.from_numpy(last_observation).float().unsqueeze(0)
        state_tensor.requires_grad = True
        state_val = stateval_network(state_tensor)

        # update critic trace
        stateval_network.zero_grad()
        state_val.backward()

        if not critic_trace:
            with torch.no_grad():
                for p in stateval_network.parameters():
                    # initialize trace
                    trace = i * p.grad
                    critic_trace.append(trace)
        else:
            with torch.no_grad():
                for i, p in enumerate(stateval_network.parameters()):
                    # decay trace and shift trace in direction of most recent gradient
                    critic_trace[i] = (
                        critic_trace[i] * CRITIC_LAMBDA * DISCOUNT_FACTOR + i * p.grad
                    )

        # get state value of next state
        new_state_tensor = torch.from_numpy(curr_observation).float().unsqueeze(0)
        new_state_val = stateval_network(new_state_tensor)

        # if terminal state, next state val is 0
        if done:
            new_state_val = torch.tensor([0]).float().unsqueeze(0)

        # calculate advantage
        advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()

        # Backpropagate policy
        policy_network.zero_grad()
        # update parameters with trace
        with torch.no_grad():
            for i, p in enumerate(policy_network.parameters()):
                new_val = p + ACTOR_LR * advantage * actor_trace[i]
                p.copy_(new_val)

        # Backpropagate value
        stateval_network.zero_grad()
        # update parameters with trace
        with torch.no_grad():
            for i, p in enumerate(stateval_network.parameters()):
                new_val = p + CRITIC_LR * advantage * critic_trace[i]
                p.copy_(new_val)

        # update I
        i *= DISCOUNT_FACTOR
        self.I[agent_id] = i

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
