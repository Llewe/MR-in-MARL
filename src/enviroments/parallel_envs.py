from pettingzoo import ParallelEnv

from src.config import env_config


def _setup_parallel_simple() -> ParallelEnv:
    from pettingzoo.mpe import simple_v3

    return simple_v3.parallel_env(
        render_mode=env_config.RENDER_MODE,
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )


def _setup_parallel_simple_tag() -> ParallelEnv:
    from pettingzoo.mpe import simple_tag_v3

    return simple_tag_v3.parallel_env(
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        render_mode=env_config.RENDER_MODE,
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )


def _setup_parallel_simple_spread() -> ParallelEnv:
    from pettingzoo.mpe import simple_spread_v3

    return simple_spread_v3.parallel_env(
        N=3,
        local_ratio=0.5,
        render_mode=env_config.RENDER_MODE,
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )
