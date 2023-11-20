from enum import Enum


class AgentType(str, Enum):
    RANDOM = "random"
    A2C = "a2c"
    DEMO_MANIPULATION_AGENT = "demo_manipulation_agent"
    DEMO_MANIPULATION_COIN = "demo_manipulation_coin"
