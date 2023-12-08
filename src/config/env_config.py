from pydantic_settings import BaseSettings


class EnvConfig(BaseSettings):
    MAX_CYCLES: int = 150
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
    WALLS: bool = True


class HarvestConfig(EnvConfig):
    PLAYERS: int = 6
    INIT_APPLES: int = 12
    REGROW_CHANCE: float = 0.001

    GRID_WIDTH: int = 18
    GRID_HEIGHT: int = 10

    MAX_CYCLES: int = 250


class PrisonersConfig(EnvConfig):
    pass


class SamaritansConfig(EnvConfig):
    pass


class StagHuntConfig(EnvConfig):
    pass


class ChickenConfig(EnvConfig):
    pass
