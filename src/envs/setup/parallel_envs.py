from pettingzoo import ParallelEnv
from torch.utils.tensorboard import SummaryWriter

from src.cfg_manager import get_cfg
from src.config.env_config import CoinGameConfig, HarvestConfig, PrisonersConfig


def _p_setup_my_coin_game(
    writer: SummaryWriter, coin_game_config: CoinGameConfig
) -> ParallelEnv:
    from src.envs.aec.my_coin_game import parallel_env

    return parallel_env(
        with_none_action=coin_game_config.WITH_NONE_ACTION,
        walls=coin_game_config.WALLS,
        max_cycles=coin_game_config.MAX_CYCLES,
        render_mode=get_cfg().get_render_mode(),
        n_players=coin_game_config.PLAYERS,
        grid_size=coin_game_config.GRID_SIZE,
        randomize_coin=True,
        allow_overlap_players=coin_game_config.ALLOW_OVERLAP_PLAYERS,
        summary_writer=writer,
    )


def _p_setup_harvest(
    writer: SummaryWriter, harvest_config: HarvestConfig
) -> ParallelEnv:
    from src.envs.aec.harvest import parallel_env

    return parallel_env(
        max_cycles=harvest_config.MAX_CYCLES,
        render_mode=get_cfg().get_render_mode(),
        n_players=harvest_config.PLAYERS,
        n_apples=harvest_config.INIT_APPLES,
        regrow_chance=harvest_config.REGROW_CHANCE,
        grid_width=harvest_config.GRID_WIDTH,
        grid_height=harvest_config.GRID_HEIGHT,
        fixed_spawn=harvest_config.FIXED_APPLE_SPAWN,
        summary_writer=writer,
    )


def _p_setup_prisoners(
    writer: SummaryWriter, prisoners_config: PrisonersConfig
) -> ParallelEnv:
    from src.envs.aec.dilemma.dilemma_pettingzoo import parallel_env

    return parallel_env(
        game="pd", max_cycles=prisoners_config.MAX_CYCLES, summary_writer=writer
    )
