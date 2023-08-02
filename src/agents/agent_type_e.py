from enum import Enum


class AgentType(str, Enum):
    RANDOM = "random"
    ACTOR_CRITIC_TD0 = "actor_critic_td0"
    ACTOR_CRITIC_TD_LAMBDA_BACKWARD_VIEW = "actor_critic_td_lambda_backward_view"
    ACTOR_CRITIC = "actor_critic"
