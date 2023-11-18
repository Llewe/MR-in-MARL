from pydantic_settings import BaseSettings


class EvalConfig(BaseSettings):
    NUM_EPISODES: int = 100
    EVAL_INTERVAL: int = 5000


eval_config = EvalConfig()
