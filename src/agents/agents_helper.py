from src.enums.agent_type_e import AgentType
from src.interfaces.agents_i import IAgents

from src import training_config

from .implementations import (
    RandomAgents,
    ActorCritic,
    DemoManipulationAgent,
)

from .implementations.a2c_examples import (
    ActorCriticTd0,
    ActorCriticTdLamdaBack,
)


def get_agents(agent_type: AgentType = training_config.AGENT_TYPE) -> IAgents:
    match agent_type:
        case AgentType.RANDOM:
            return RandomAgents()

        case AgentType.ACTOR_CRITIC:
            return ActorCritic()

        case AgentType.DEMO_MANIPULATION_AGENT:
            return DemoManipulationAgent()

        case AgentType.ACTOR_CRITIC_TD0:
            return ActorCriticTd0()

        case AgentType.ACTOR_CRITIC_TD_LAMBDA_BACKWARD_VIEW:
            return ActorCriticTdLamdaBack()
