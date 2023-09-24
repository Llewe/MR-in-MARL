from pydantic_settings import BaseSettings

from src.enums.agent_type_e import AgentType


class TrainingConfig(BaseSettings):
    EPISODES: int = 2000
    EPOCHS: int = 1
    AGENT_TYPE: AgentType = AgentType.RANDOM


training_config = TrainingConfig()
