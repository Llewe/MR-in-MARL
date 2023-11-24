from pettingzoo import AECEnv
from pettingzoo.utils import parallel_to_aec
from shimmy.meltingpot_compatibility import MeltingPotCompatibilityV0
from torch.utils.tensorboard import SummaryWriter

from src.cfg_manager import get_cfg
from src.config.env_config import (
    ChickenConfig,
    CoinGameConfig,
    EnvConfig,
    PrisonersConfig,
    SamaritansConfig,
    SimpleAdversaryConfig,
    SimpleSpreadConfig,
    SimpleTagConfig,
    StagHuntConfig,
)


def _setup_simple(env_config: EnvConfig) -> AECEnv:
    from pettingzoo.mpe import simple_v3

    return simple_v3.env(
        render_mode=get_cfg().get_render_mode(),
        max_cycles=env_config.MAX_CYCLES,
        continuous_actions=env_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_tag(tag_config: SimpleTagConfig) -> AECEnv:
    import pettingzoo.mpe.simple_tag_v3 as simple_tag_v3

    return simple_tag_v3.env(
        num_good=tag_config.NUM_GOOD,
        num_adversaries=tag_config.NUM_ADVERSARIES,
        num_obstacles=tag_config.NUM_OBSTACLES,
        render_mode=get_cfg().get_render_mode(),
        max_cycles=tag_config.MAX_CYCLES,
        continuous_actions=tag_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_spread(spread_config: SimpleSpreadConfig) -> AECEnv:
    from pettingzoo.mpe import simple_spread_v3

    return simple_spread_v3.env(
        N=spread_config.NUM_AGENTS,
        local_ratio=spread_config.LOCAL_RATIO,
        render_mode=get_cfg().get_render_mode(),
        max_cycles=spread_config.MAX_CYCLES,
        continuous_actions=spread_config.CONTINUOUS_ACTIONS,
    )


def _setup_simple_adversary(adversary_config: SimpleAdversaryConfig) -> AECEnv:
    from pettingzoo.mpe import simple_adversary_v3

    return simple_adversary_v3.env(
        N=adversary_config.NUM_AGENTS,
        render_mode=get_cfg().get_render_mode(),
        max_cycles=adversary_config.MAX_CYCLES,
        continuous_actions=adversary_config.CONTINUOUS_ACTIONS,
    )


def _setup_coin_game(writer: SummaryWriter, coin_game_config: CoinGameConfig) -> AECEnv:
    from src.envs.aec.coin_game import raw_env

    return raw_env(
        render_mode=get_cfg().get_render_mode(),
        nb_players=coin_game_config.PLAYERS,
        grid_size=coin_game_config.GRID_SIZE,
        max_cycles=coin_game_config.MAX_CYCLES,
        randomize_coin=True,
        summary_writer=writer,
        allow_overlap_players=coin_game_config.ALLOW_OVERLAP_PLAYERS,
    )


def _setup_my_coin_game(
    writer: SummaryWriter, coin_game_config: CoinGameConfig
) -> AECEnv:
    from src.envs.aec.my_coin_game import CoinGame

    return CoinGame(
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


def _setup_prisoners(
    writer: SummaryWriter, prisoners_config: PrisonersConfig
) -> AECEnv:
    from src.envs.aec.dilemma.dilemma_pettingzoo import raw_env

    return raw_env(game="pd", max_cycles=prisoners_config.MAX_CYCLES)


def _setup_samaritans(
    writer: SummaryWriter, samaritans_config: SamaritansConfig
) -> AECEnv:
    from src.envs.aec.dilemma.dilemma_pettingzoo import raw_env

    return raw_env(game="sd", max_cycles=samaritans_config.MAX_CYCLES)


def _setup_stag_hunt(writer: SummaryWriter, stag_hunt_config: StagHuntConfig) -> AECEnv:
    from src.envs.aec.dilemma.dilemma_pettingzoo import raw_env

    return raw_env(game="stag", max_cycles=stag_hunt_config.MAX_CYCLES)


def _setup_chicken(writer: SummaryWriter, chicken_config: ChickenConfig) -> AECEnv:
    from src.envs.aec.dilemma.dilemma_pettingzoo import raw_env

    return raw_env(game="chicken", max_cycles=chicken_config.MAX_CYCLES)


def _setup_melting_pod(name: str, env_config: EnvConfig) -> AECEnv:
    return parallel_to_aec(
        MeltingPotCompatibilityV0(
            substrate_name=name,
            render_mode=get_cfg().get_render_mode(),
            max_cycles=env_config.MAX_CYCLES,
        )
    )
