from pydantic_settings import BaseSettings


class EvalConfig(BaseSettings):
    NUM_EPISODES: int = 500
    EVAL_INTERVAL: int = 500


eval_config = EvalConfig()
