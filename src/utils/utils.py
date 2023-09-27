import os.path
from os import makedirs
from os.path import join, dirname, realpath

from src.config.base_ctrl_config import BaseCtrlConfig
from src.config.env_config import BaseEnvConfig
from src.enums import AgentType, EnvType


def get_env_dir() -> str:
    base_config = BaseEnvConfig()

    data_dir = join(
        dirname(dirname(realpath(__file__))),
        f"../resources",
        base_config.ENV_NAME.value,
        base_config.ENV_TAG,
    )
    if not os.path.exists(data_dir):
        makedirs(data_dir)
    return data_dir


def get_ctrl_dir() -> str:
    base_ctrl_config = BaseCtrlConfig()

    data_dir = get_env_dir()
    data_dir = join(
        data_dir,
        base_ctrl_config.AGENT_TYPE.value,
        f"{base_ctrl_config.START_TIME} - {base_ctrl_config.AGENT_TAG}",
    )
    if not os.path.exists(data_dir):
        makedirs(data_dir)
    return data_dir


def get_model_storage(episode: int) -> str:
    return join(
        get_ctrl_dir(),
        f"episode-{episode}",
    )


def create_run_name(env_type: EnvType, agent_type: AgentType, tag: str) -> str:
    base_ctrl_config = BaseCtrlConfig()

    return f"{env_type.value}/{agent_type.value}/{base_ctrl_config.START_TIME} - {base_ctrl_config.AGENT_TAG}"


def get_log_folder(run_name: str) -> str:
    file = join(
        dirname(dirname(realpath(__file__))),
        "../resources/tensorboard",
        run_name,
    )
    return file
