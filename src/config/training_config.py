from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    EPISODES: int = 10
    EPOCHS: int = 50000


training_config = TrainingConfig()
