from enum import Enum

from pydantic_settings import BaseSettings

from src.enums import EnvType
from pydantic import Extra


class CtrlConfig(BaseSettings):
    NAME: str = "IPD-generic-test"

    class Config:
        extra = Extra.allow


class ACConfig(CtrlConfig):
    ACTOR_LR: float = 0.001
    CRITIC_LR: float = 0.001
    ACTOR_HIDDEN_UNITS: int = 64
    CRITIC_HIDDEN_UNITS: int = 64
    DISCOUNT_FACTOR: float = 0.99

    CLIP_NORM: float = 1.0

    REWARD_NORMALIZATION: bool = False

    EPSILON_INIT: float = 0.0
    EPSILON_MIN: float = 0.0
    EPSILON_DECAY: float = 3.0e-05


class MateConfig(ACConfig):
    class Mode(str, Enum):
        STATIC_MODE = "static"
        TD_ERROR_MODE = "td_error"
        VALUE_DECOMPOSE_MODE = "value_decompose"

    class DefectMode(str, Enum):
        NO_DEFECT = "no_defect"
        DEFECT_ALL = (
            "defect_all"  # Does not send or receive any acknowledgment messages
        )
        DEFECT_RESPONSE = "defect_response"  # Sends acknowledgment requests but does not respond to incoming requests
        DEFECT_RECEIVE = "defect_receive"  # Sends acknowledgment requests but does not receive any responses
        DEFECT_SEND = "defect_send"  # Receives acknowledgment requests but does send any requests itself

    MODE: Mode = Mode.TD_ERROR_MODE
    DEFECT_MODE: DefectMode = DefectMode.NO_DEFECT
    TOKEN_VALUE: float = 1.0


class LolaPGConfig(ACConfig):
    SECOND_ORDER_LR: float = 1


class GiftingConfig(ACConfig):
    class Mode(str, Enum):
        ZERO_SUM = "zero_sum"
        FIXED_BUDGET = "fixed_budget"
        REPLENISHABLE_BUDGET = "replenishable_budget"

    class ActionMode(str, Enum):
        RANDOM = "random"
        NO_ACTION = "no_action"

    @staticmethod
    def index_from_env(env_type: EnvType):
        if env_type == EnvType.P_COIN_GAME:
            return 4
        elif env_type == EnvType.P_HARVEST:
            return 4

    GIFT_REWARD: float = 1.0
    GIFT_MODE: Mode = Mode.REPLENISHABLE_BUDGET

    # Only used for FIXED_BUDGET and REPLENISHABLE_BUDGET
    GIFT_BUDGET: float = 10.0

    ACTION_MODE: ActionMode = ActionMode.RANDOM
    ENV_USED: EnvType = EnvType.P_COIN_GAME
    ENV_NONE_ACTION_INDEX: int = index_from_env(
        ENV_USED
    )  # The action is the last one in the action space


class MaConfig(ACConfig):
    MANIPULATION_AMOUNT: float = 2.0


class MaACConfig(ACConfig):
    MANIPULATION_AMOUNT: float = 3.0


class MaMATEConfig(MateConfig):
    MANIPULATION_AMOUNT: float = 1.0


class DemoMaCoinConfig(ACConfig):
    MANIPULATION_AMOUNT: float = 0.2


class DemoMaConfig(ACConfig):
    MANIPULATION_AMOUNT: float = 0.1


if __name__ == "__main__":
    # Calculate Decay rate:
    epoch_reached_min = 2000

    a2c_config = ACConfig()

    decay = (a2c_config.EPSILON_INIT - a2c_config.EPSILON_MIN) / epoch_reached_min
    print(
        f"With epsilon start ({a2c_config.EPSILON_INIT}) epsilon min "
        f"({a2c_config.EPSILON_MIN}) will be reached after {epoch_reached_min} "
        f"epochs with the decay rate of {decay}"
    )
    print(
        f"EPSILON_INIT={a2c_config.EPSILON_INIT} EPSILON_DECAY={decay} EPSILON_MIN={a2c_config.EPSILON_MIN}"
    )
