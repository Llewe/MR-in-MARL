from pydantic_settings import BaseSettings

from src.enums.agent_type_e import AgentType


class TrainingConfig(BaseSettings):
    EPISODES: int = 10000
    EPOCHS: int = 1
    AGENT_TYPE: AgentType = AgentType.ACTOR_CRITIC


training_config = TrainingConfig()
