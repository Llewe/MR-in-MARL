import names

from pydantic_settings import BaseSettings


class LogConfig(BaseSettings):
    LOG_LEVEL: str = "INFO"

    NAME_TAG: str = names.get_first_name()


log_config = LogConfig()
