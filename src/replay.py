import logging
from os.path import join, dirname, realpath

import pygame
from pettingzoo import AECEnv

from agents.agents_helper import get_agents
from src import replay_config, env_config, log_config, build_env
from src.config import pygame_config
from src.interfaces.agents_i import IAgents

logging.basicConfig(level=logging.getLevelName(log_config.LOG_LEVEL))


def replay(aec_env: AECEnv, agent: IAgents, timeout=replay_config.TIMEOUT) -> None:
    i: int = 0
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

    RUN_NAME = f"{replay_config.ENV_NAME}/{replay_config.ENV_TAG}/{replay_config.AGENT_TYPE}/{replay_config.EXPERIMENT_NAME}"

    logging.info(f"Loading agents from {RUN_NAME}")
    model_storage = join(
        dirname(dirname(realpath(__file__))),
        "resources",
        RUN_NAME,
        f"episode-{replay_config.EPISODE}",
    )

    agents.load(model_storage)

    return agents


if __name__ == "__main__":
    logging.info("Starting MR-in-MARL")

    logging.info(f"Loading agents from {replay_config.ENV_NAME}")

    pygame_config.RENDER_MODE = "human"
    pygame_config.RENDER_FPS = 60
    env_config.MAX_CYCLES = replay_config.STEPS
    env: AECEnv = build_env(replay_config.ENV_NAME)

    logging.info(f"Loading agents from {replay_config.AGENT_TYPE}")
    agents: IAgents = load_agents(env)

    logging.info("Starting replay")
    replay(env, agents)
