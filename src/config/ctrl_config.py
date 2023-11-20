from pydantic_settings import BaseSettings


class CtrlConfig(BaseSettings):
    pass


class A2cConfig(CtrlConfig):
    ACTOR_LR: float = 0.0001
    CRITIC_LR: float = 0.0001
    ACTOR_HIDDEN_UNITS: int = 64
    CRITIC_HIDDEN_UNITS: int = 64
    DISCOUNT_FACTOR: float = 0.95

    CLIP_NORM: float = 1

    REWARD_NORMALIZATION: bool = False

    EPSILON_INIT: float = 0.8
    EPSILON_MIN: float = 0.1
    EPSILON_DECAY: float = 1.5e-05


class DemoMaCoinConfig(A2cConfig):
    MANIPULATION_AMOUNT: float = 0.2


class DemoMaConfig(A2cConfig):
    MANIPULATION_AMOUNT: float = 0.1


if __name__ == "__main__":
    # Calculate Decay rate:
    epoch_reached_min = 45000

    a2c_config = A2cConfig()

    decay = (a2c_config.EPSILON_INIT - a2c_config.EPSILON_MIN) / epoch_reached_min
    print(
        f"With epsilon start ({a2c_config.EPSILON_INIT}) epsilon min "
        f"({a2c_config.EPSILON_MIN}) will be reached after {epoch_reached_min} "
        f"epochs with the decay rate of {decay}"
    )
