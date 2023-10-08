from pydantic_settings import BaseSettings

from src.enums.env_type_e import EnvType


class BaseEnvConfig(BaseSettings):
    ENV_NAME: EnvType = EnvType.COIN_GAME
    ENV_TAG: str = "4p 5x5"


class EnvConfig(BaseSettings):
    MAX_CYCLES: int = 25
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
    PLAYERS: int = 4
    GRID_SIZE: int = 5
    ALLOW_OVERLAP_PLAYERS: bool = True


class PrisonersConfig(EnvConfig):
    PLAYERS: int = 1
    GRID_SIZE: int = 2
    ALLOW_OVERLAP_PLAYERS: bool = True


class SamaritansConfig(EnvConfig):
    pass


class StagHuntConfig(EnvConfig):
    pass


class ChickenConfig(EnvConfig):
    pass


env_config = EnvConfig()
