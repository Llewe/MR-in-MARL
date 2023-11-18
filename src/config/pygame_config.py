from pydantic_settings import BaseSettings


class PyGameConfig(BaseSettings):
    RENDER_MODE: str = ""
    RENDER_FPS: int = 30


pygame_config = PyGameConfig()


def update_pygame_config(render_mode: str, render_fps: int):
    global pygame_config
    pygame_config.RENDER_MODE = render_mode
    pygame_config.RENDER_FPS = render_fps
