import os.path
from os import makedirs
from os.path import dirname, join, realpath

from pydantic_settings import BaseSettings

from src.enums import AgentType, EnvType


class PyGameConfig(BaseSettings):
    RENDER_MODE: str = "human"
    RENDER_FPS: int = 30


class LogConfig(BaseSettings):
    LOG_LEVEL: str = "INFO"


class ExpConfig(BaseSettings):
    EXP_UNIQUE_NAME: str
    AGENT_TYPE: AgentType

    ENV_NAME: EnvType
    ENV_TAG: str


class CfgManager:
    pygame_config: PyGameConfig
    log_config: LogConfig
    exp_config: ExpConfig

    dir_env: str = ""
    dir_ctrl: str = ""

    def __init__(self, exp_config: ExpConfig):
        self.pygame_config = PyGameConfig()
        self.log_config = LogConfig()
        self.exp_config = exp_config

        self.dir_env = self._setup_env_dir()
        self.dir_ctrl = self._setup_ctrl_dir()

    def _setup_env_dir(self) -> str:
        data_dir = join(
            dirname(dirname(realpath(__file__))),
            f"resources",
            self.exp_config.ENV_NAME.value,
            self.exp_config.ENV_TAG,
        )
        if not os.path.exists(data_dir):
            makedirs(data_dir)

        return data_dir

    def _setup_ctrl_dir(self) -> str:
        data_dir = join(
            self.dir_env,
            self.exp_config.AGENT_TYPE.value,
            self.exp_config.EXP_UNIQUE_NAME,
        )
        if not os.path.exists(data_dir):
            makedirs(data_dir)
        return data_dir

    def get_env_dir(self) -> str:
        return self.dir_env

    def get_ctrl_dir(self) -> str:
        return self.dir_ctrl

    def get_model_storage(self, epoch: int) -> str:
        return join(
            self.dir_ctrl,
            f"epoch-{epoch}",
        )

    def get_render_mode(self) -> str:
        return self.pygame_config.RENDER_MODE

    def get_render_fps(self) -> int:
        return self.pygame_config.RENDER_FPS

    def get_log_level(self) -> str:
        return self.log_config.LOG_LEVEL

    def update_pygame_config(self, render_mode: str, render_fps: int):
        self.pygame_config.RENDER_MODE = render_mode
        self.pygame_config.RENDER_FPS = render_fps


_cfg_manager: CfgManager


def get_cfg() -> CfgManager:
    global _cfg_manager
    return _cfg_manager


def set_cfg(cfg_manager: CfgManager) -> None:
    global _cfg_manager
    _cfg_manager = cfg_manager
