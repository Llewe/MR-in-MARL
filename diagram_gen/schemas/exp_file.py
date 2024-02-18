from typing import Dict, Optional

from pydantic import BaseModel

from src.config.ctrl_config import CtrlConfig


class ExpFile(BaseModel):
    path: str
    agent_type: str
    diagram_data: Optional[Dict]

    cfg: Optional[CtrlConfig]
