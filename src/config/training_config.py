from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    EPISODES: int = 2000
    EPOCHS: int = 1


training_config = TrainingConfig()
