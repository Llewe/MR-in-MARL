from pettingzoo import AECEnv
from torch.utils.tensorboard import SummaryWriter

from src.config.pygame_config import pygame_config
from src.config.env_config import (
    SimpleTagConfig,
    SimpleSpreadConfig,
    EnvConfig,
    SimpleAdversaryConfig,
    CoinGameConfig,
)


def _setup_simple(env_config: EnvConfig) -> AECEnv:
    import pettingzoo.mpe.simple_v3 as simple_v3

    return simple_v3.env(
        render_mode=pygame_config.RENDER_MODE,
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_tag(tag_config: SimpleTagConfig) -> AECEnv:
    import pettingzoo.mpe.simple_tag_v3 as simple_tag_v3

    return simple_tag_v3.env(
        num_good=tag_config.NUM_GOOD,
        num_adversaries=tag_config.NUM_ADVERSARIES,
        num_obstacles=tag_config.NUM_OBSTACLES,
        render_mode=pygame_config.RENDER_MODE,
        max_cycles=tag_config.MAX_CYCLES,
        continuous_actions=tag_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_spread(spread_config: SimpleSpreadConfig) -> AECEnv:
    from pettingzoo.mpe import simple_spread_v3

    return simple_spread_v3.env(
        N=spread_config.NUM_AGENTS,
        local_ratio=spread_config.LOCAL_RATIO,
        render_mode=pygame_config.RENDER_MODE,
        max_cycles=spread_config.MAX_CYCLES,
        continuous_actions=spread_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_adversary(adversary_config: SimpleAdversaryConfig) -> AECEnv:
    from pettingzoo.mpe import simple_adversary_v3

    return simple_adversary_v3.env(
        N=adversary_config.NUM_AGENTS,
        render_mode=pygame_config.RENDER_MODE,
        max_cycles=adversary_config.MAX_CYCLES,
        continuous_actions=adversary_config.CONTINUOUS_ACTIONS,
    )


def _setup_coin_game(writer: SummaryWriter, coin_game_config: CoinGameConfig) -> AECEnv:
    from src.envs.aec.coin_game import raw_env
    return raw_env(
        render_mode=pygame_config.RENDER_MODE,
        nb_players=coin_game_config.PLAYERS,
        grid_size=coin_game_config.GRID_SIZE,
        max_cycles=coin_game_config.MAX_CYCLES,
        randomize_coin=True,
        summary_writer=writer,
        allow_overlap_players=coin_game_config.ALLOW_OVERLAP_PLAYERS,
    )
