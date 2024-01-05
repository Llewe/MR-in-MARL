import logging

import pygame
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils.env import AgentID, ActionType

from src import build_env
from src.controller.utils.agents_helper import get_agents
from src.cfg_manager import CfgManager, get_cfg, set_cfg
from src.config.replay_config import ReplayConfig
from src.interfaces.controller_i import IController
from torch.utils.tensorboard import SummaryWriter

_replay_config = ReplayConfig()


def replay(
    aec_env: AECEnv, agent: IController, timeout=_replay_config.TIMEOUT, logging=False
) -> None:
    i: int = 0
    pygame.init()
    aec_env.reset()
    while True:
        i += 1

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
        options = None
        if logging:
            options = {
                "write_log": True,
                "epoch": i,
                "tag": "eval",
            }
        aec_env.reset()

        logging.info(f"another one - {i}")


def replay_parallel(
    env: ParallelEnv, agents: IController, timeout=_replay_config.TIMEOUT, logging=False
) -> None:
    i: int = 0
    if get_cfg().get_render_mode() != "":
        pygame.init()
    timestep: int = 0

    observations, infos = env.reset(
        options={
            "write_log": False,
            "epoch": i,
            "tag": "eval",
        }
    )
    max_iterations: int = 1
    iterations: int = 0
    while iterations < max_iterations:
        iterations += 1
        while env.agents:
            if get_cfg().get_render_mode() != "":
                pygame.event.get()  # so that the window doesn't freeze

            timestep += 1
            actions: dict[AgentID, ActionType] = agents.act_parallel(
                observations, explore=True
            )

            new_observations, _, _, _, _ = env.step(actions)
            observations = new_observations

            if timeout > 0:
                pygame.time.wait(timeout)

        observations, infos = env.reset(
            options={
                "write_log": True,
                "epoch": i,
                "tag": "train",
                "heatmap": True,
            }
        )
        i += 1


def replay_with_logs(
    aec_env: AECEnv,
    agent: IController,
    timeout=_replay_config.TIMEOUT,
) -> None:
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


def load_agents(aec_env: AECEnv) -> IController:
    agents: IController = get_agents()
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

    with_logs = True

    env: ParallelEnv | AECEnv
    agents: IController

    if not with_logs:
        get_cfg().update_pygame_config(
            render_mode="human",
            render_fps=60,
        )

        env = build_env(get_cfg().exp_config.ENV_NAME)

        logging.info(f"Loading agents from {get_cfg().exp_config.AGENT_TYPE}")
        agents = load_agents(env)

        logging.info("Starting replay")
        replay(env, agents)
    else:
        get_cfg().update_pygame_config(
            render_mode="human",
            render_fps=60,
        )
        agent_dir: str = get_cfg().get_ctrl_dir()

        logging.info(f"TensorBoard -> Logging to {agent_dir}")

        with SummaryWriter(log_dir=agent_dir) as writer:
            env = build_env(get_cfg().exp_config.ENV_NAME, writer)

            logging.info(f"Loading agents from {get_cfg().exp_config.AGENT_TYPE}")
            agents = load_agents(env)

            logging.info("Starting replay")
            if isinstance(env, ParallelEnv):
                replay_parallel(env, agents, logging=True)
            else:
                replay(env, agents, logging=True)
