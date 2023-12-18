from enum import Enum

from pydantic_settings import BaseSettings


class CtrlConfig(BaseSettings):
    pass


class A2cConfig(CtrlConfig):
    ACTOR_LR: float = 0.001
    CRITIC_LR: float = 0.001
    ACTOR_HIDDEN_UNITS: int = 64
    CRITIC_HIDDEN_UNITS: int = 64
    DISCOUNT_FACTOR: float = 0.99

    CLIP_NORM: float = 1

    REWARD_NORMALIZATION: bool = False

    EPSILON_INIT: float = 0.8
    EPSILON_MIN: float = 0.1
    EPSILON_DECAY: float = 0.0002


class MateConfig(A2cConfig):
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
    TOKEN_VALUE: float = 1


class GiftingConfig(A2cConfig):
    class Mode(str, Enum):
        ZERO_SUM_MODE = "zero_sum"
        BUDGET_MODE = "budget"

    GIFT_REWARD: float = 1
    GIFT_MODE: Mode = Mode.ZERO_SUM_MODE


class DemoMaCoinConfig(A2cConfig):
    MANIPULATION_AMOUNT: float = 0.2


class DemoMaConfig(A2cConfig):
    MANIPULATION_AMOUNT: float = 0.1


if __name__ == "__main__":
    # Calculate Decay rate:
    epoch_reached_min = 35000

    a2c_config = A2cConfig()

    decay = (a2c_config.EPSILON_INIT - a2c_config.EPSILON_MIN) / epoch_reached_min
    print(
        f"With epsilon start ({a2c_config.EPSILON_INIT}) epsilon min "
        f"({a2c_config.EPSILON_MIN}) will be reached after {epoch_reached_min} "
        f"epochs with the decay rate of {decay}"
    )
    print(
        f"EPSILON_INIT={a2c_config.EPSILON_INIT} EPSILON_DECAY={decay} EPSILON_MIN={a2c_config.EPSILON_MIN}"
    )
