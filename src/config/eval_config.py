from pydantic_settings import BaseSettings


class EvalConfig(BaseSettings):
    NUM_EPISODES: int = 20
    EVAL_INTERVAL: int = 2000


eval_config = EvalConfig()
