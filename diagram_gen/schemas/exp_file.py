from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ExpFile:
    path: str
    agent_type: str
    diagram_data: Optional[Dict]
