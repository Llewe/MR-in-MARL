from pydantic_settings import BaseSettings


from src.enums.env_type_e import EnvType


class EnvConfig(BaseSettings):
    ENV_NAME: EnvType = EnvType.SIMPLE
    RENDER_MODE: str = ""
    RENDER_FPS: int = 30

    MAX_CYCLES: int = 25
    CONTINUOUS_ACTIONS: bool = False


env_config = EnvConfig()
