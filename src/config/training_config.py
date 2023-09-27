from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    EPISODES: int = 40000
    EPOCHS: int = 1


training_config = TrainingConfig()
