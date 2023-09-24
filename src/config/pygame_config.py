from pydantic_settings import BaseSettings


class PyGameConfig(BaseSettings):
    RENDER_MODE: str = ""
    RENDER_FPS: int = 30


pygame_config = PyGameConfig()
