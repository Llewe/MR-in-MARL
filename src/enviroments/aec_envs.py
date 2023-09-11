from pettingzoo import AECEnv

from src.config import env_config


def _setup_simple() -> AECEnv:
    import pettingzoo.mpe.simple_v3 as simple_v3

    return simple_v3.env(
        render_mode=env_config.RENDER_MODE,
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_tag() -> AECEnv:
    import pettingzoo.mpe.simple_tag_v3 as simple_tag_v3

    return simple_tag_v3.env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        render_mode=env_config.RENDER_MODE,
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_spread() -> AECEnv:
    from pettingzoo.mpe import simple_spread_v3

    return simple_spread_v3.env(
        N=3,
        local_ratio=0.5,
        render_mode=env_config.RENDER_MODE,
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_adversary() -> AECEnv:
    from pettingzoo.mpe import simple_adversary_v3

    return simple_adversary_v3.env(
        N=2,
        render_mode=env_config.RENDER_MODE,
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )
