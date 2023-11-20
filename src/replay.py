import logging

import pygame
from pettingzoo import AECEnv

from src import build_env
from src.agents.utils.agents_helper import get_agents
from src.cfg_manager import CfgManager, get_cfg, set_cfg
from src.config.replay_config import ReplayConfig
from src.interfaces.agents_i import IAgents

_replay_config = ReplayConfig()


def replay(aec_env: AECEnv, agent: IAgents, timeout=_replay_config.TIMEOUT) -> None:
    i: int = 0
    pygame.init()
    while True:
        i += 1
        aec_env.reset()

        for agent_id in aec_env.agent_iter():
            pygame.event.get()  # so that the window doesn't freeze
            observation, reward, termination, truncation, info = aec_env.last()

            if termination or truncation:
                action = None
            else:
                action = agent.act(
                    agent_id=agent_id, observation=observation, explore=False
                )

            aec_env.step(action)

            if timeout > 0:
                pygame.time.wait(timeout)

        logging.info(f"another one - {i}")


def load_agents(aec_env: AECEnv) -> IAgents:
    agents: IAgents = get_agents()
    agents.init_agents(
        action_space={a: aec_env.action_space(a) for a in aec_env.possible_agents},
        observation_space={
            a: aec_env.observation_space(a) for a in aec_env.possible_agents
        },
    )

    agents.load(get_cfg().get_model_storage(_replay_config.EPOCH))

    return agents


if __name__ == "__main__":
    set_cfg(CfgManager(_replay_config))
    logging.basicConfig(level=logging.getLevelName(get_cfg().get_log_level()))
    logging.info("Starting MR-in-MARL")

    logging.info(f"Loading agents from {_replay_config.ENV_NAME}")

    get_cfg().update_pygame_config(
        render_mode="human",
        render_fps=60,
    )

    env: AECEnv = build_env(get_cfg().exp_config.ENV_NAME)

    logging.info(f"Loading agents from {get_cfg().exp_config.AGENT_TYPE}")
    agents: IAgents = load_agents(env)

    logging.info("Starting replay")
    replay(env, agents)
