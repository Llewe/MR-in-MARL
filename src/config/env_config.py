from pydantic_settings import BaseSettings

from src.enums.env_type_e import EnvType


class BaseEnvConfig(BaseSettings):
    ENV_NAME: EnvType = EnvType.COIN_GAME
    ENV_TAG: str = "2p 3x3"


class EnvConfig(BaseSettings):
    MAX_CYCLES: int = 60
    CONTINUOUS_ACTIONS: bool = False


class SimpleTagConfig(EnvConfig):
    # Simple Tag
    NUM_GOOD: int = 1
    NUM_ADVERSARIES: int = 3
    NUM_OBSTACLES: int = 2


class SimpleSpreadConfig(EnvConfig):
    # Simple Spread
    NUM_AGENTS: int = 3
    LOCAL_RATIO: float = 0.5


class SimpleAdversaryConfig(EnvConfig):
    # Simple Adversary
    NUM_AGENTS: int = 2


class CoinGameConfig(EnvConfig):
    PLAYERS: int = 2
    GRID_SIZE: int = 3
    ALLOW_OVERLAP_PLAYERS: bool = True


env_config = EnvConfig()
