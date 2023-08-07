import config
from agents.agent_type_e import AgentType
from agents.agents_i import IAgents
from agents.old.actor_critic import ActorCritic
from config import actor_critic_config, training_config
from pettingzoo import ParallelEnv, AECEnv


def setup_agents_old(env: ParallelEnv | AECEnv) -> ActorCritic:
    return ActorCritic(
        env.possible_agents,
        [env.observation_space(a) for a in env.possible_agents],
        [env.action_space(a) for a in env.possible_agents],
        actor_hidden_units=actor_critic_config.ACTOR_HIDDEN_UNITS,
        critic_hidden_units=actor_critic_config.CRITIC_HIDDEN_UNITS,
        actor_lr=actor_critic_config.ACTOR_LR,
        critic_lr=actor_critic_config.CRITIC_LR,
    )


def get_agents(agent_type: AgentType = training_config.AGENT_TYPE) -> IAgents:
    match agent_type:
        case AgentType.RANDOM:
            from agents.implementations.random_agents import RandomAgents

            return RandomAgents()

        case AgentType.ACTOR_CRITIC:
            from agents.implementations.actor_critic import ActorCritic

            return ActorCritic()

        case AgentType.ACTOR_CRITIC_TD0:
            from agents.implementations.actor_critic_td0 import ActorCriticTd0

            return ActorCriticTd0()
        case AgentType.ACTOR_CRITIC_TD_LAMBDA_BACKWARD_VIEW:
            from agents.implementations.actor_critic_td_lamda_back import (
                ActorCriticTdLamdaBack,
            )

            return ActorCriticTdLamdaBack()
