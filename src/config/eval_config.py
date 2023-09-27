from pydantic_settings import BaseSettings


class EvalConfig(BaseSettings):
    NUM_EPISODES: int = 300
    EVAL_INTERVAL: int = 40000


eval_config = EvalConfig()
