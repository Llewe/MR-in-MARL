from datetime import datetime
from time import time
from pydantic_settings import BaseSettings

import names

from src.enums import AgentType


class BaseCtrlConfig(BaseSettings):
    AGENT_TYPE: AgentType = AgentType.A2C
    AGENT_TAG: str = names.get_first_name()

    START_TIME: str = datetime.fromtimestamp(time()).isoformat(timespec="seconds")
