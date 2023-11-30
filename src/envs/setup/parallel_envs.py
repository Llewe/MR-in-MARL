from pettingzoo import ParallelEnv
from torch.utils.tensorboard import SummaryWriter

from src.cfg_manager import get_cfg
from src.config.env_config import CoinGameConfig


def _p_setup_my_coin_game(
    writer: SummaryWriter, coin_game_config: CoinGameConfig
) -> ParallelEnv:
    from src.envs.aec.my_coin_game import parallel_env

    return parallel_env(
        with_none_action=True,
        walls=coin_game_config.WALLS,
        max_cycles=coin_game_config.MAX_CYCLES,
        render_mode=get_cfg().get_render_mode(),
        n_players=coin_game_config.PLAYERS,
        grid_size=coin_game_config.GRID_SIZE,
        randomize_coin=True,
        allow_overlap_players=coin_game_config.ALLOW_OVERLAP_PLAYERS,
        summary_writer=writer,
    )
