from pydantic_settings import BaseSettings

from src.enums.agent_type_e import AgentType


class TrainingConfig(BaseSettings):
    EPISODES: int = 100
    EPOCHS: int = 1
    AGENT_TYPE: AgentType = AgentType.DEMO_MANIPULATION_AGENT


training_config = TrainingConfig()
