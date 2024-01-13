import logging
from collections import defaultdict
from typing import Callable, Union

import numpy
import numpy as np
import pygame
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.env import ActionType, AgentID, ObsType
from torch.utils.tensorboard import SummaryWriter

from src.cfg_manager import CfgManager, get_cfg, set_cfg
from src.config.training_config import TrainingConfig
from src.controller.utils.agents_helper import get_agents
from src.controller_ma.utils.ma_setup_helper import (
    get_global_obs,
    get_ma_controller,
    get_metrics,
    get_obs_space,
    get_wanted_metrics,
)
from src.envs import build_env
from src.interfaces.controller_i import IController
from src.interfaces.ma_controller_i import IMaController
from src.utils.loggers.obs_logger import IObsLogger
from src.utils.loggers.simple_env_logger import SimpleEnvLogger
from src.utils.loggers.util_logger import log_efficiency

_training_config = TrainingConfig()


def _train_epoch(
    ma_controller: IMaController,
    controller: IController,
    env: Union[AECEnv, ParallelEnv],
    current_epoch: int,
    writer: SummaryWriter,
    obs_logger: IObsLogger,
    parallel: bool = False,
) -> None:
    controller.epoch_started(current_epoch)
    ma_controller.epoch_started(current_epoch)

    # For coin game this resets the history.Important in case e.g. eval is done in between
    env.reset(options={"history_reset": True})

    obs_logger.clear_buffer()

    epoch_rewards: dict[AgentID, list[float]] = {a: [] for a in env.agents}

    for episode in range(1, _training_config.EPISODES + 1):
        logging.info(
            f"Epoch {current_epoch}/{_training_config.EPOCHS}"
            f" Episode: {episode}/{_training_config.EPISODES}"
        )
        if parallel:
            episode_reward: dict[AgentID, float] = _train_parallel_episode(
                ma_controller,
                controller,
                env,
                current_epoch,
                episode,
                writer,
                obs_logger,
            )
        else:
            raise NotImplementedError("Only parallel envs are supported")

        for agent_id in episode_reward:
            epoch_rewards[agent_id].append(episode_reward[agent_id])

    # by resetting some envs (e.g. coin game) will log some env specific data
    env.reset(
        options={
            "write_log": True,
            "epoch": current_epoch,
            "tag": "train",
        }
    )

    for agent_id, epoch_reward in epoch_rewards.items():
        writer.add_scalar(
            f"train-real-rewards/{agent_id}", np.mean(epoch_reward), current_epoch
        )
    log_efficiency(
        current_epoch, "train", writer, epoch_rewards, _training_config.EPISODES
    )

    obs_logger.log_epoch(current_epoch, "train")

    controller.epoch_finished(current_epoch, "train")
    ma_controller.epoch_finished(current_epoch, "train")


def _train_parallel_episode(
    ma_controller: IMaController,
    controller: IController,
    env: ParallelEnv,
    current_epoch: int,
    current_episode: int,
    writer: SummaryWriter,
    obs_logger: IObsLogger,
) -> dict[AgentID, float]:
    timestep: int = 0

    observations, infos = env.reset(
        options={
            "write_log": False,
            "epoch": current_epoch,
            "tag": "train",
        }
    )
    episode_reward: dict[AgentID, float] = {a: 0 for a in env.possible_agents}

    ma_controller.episode_started(current_episode)

    obs_callback: Callable[[], ObsType] | None = get_global_obs(
        _training_config.MA_MODE, env
    )
    metrics = get_wanted_metrics(_training_config.MA_MODE)

    controller.episode_started(current_episode)
    while env.agents:
        if get_cfg().get_render_mode() != "":
            pygame.event.get()  # so that the window doesn't freeze

        timestep += 1
        actions: dict[AgentID, ActionType] = controller.act_parallel(
            observations, explore=True
        )

        for agent_id, observation in observations.items():
            obs_logger.add_observation(agent_id, observation)

        new_observations, rewards, terminations, truncations, infos = env.step(actions)

        current_metrics = (
            get_metrics(_training_config.MA_MODE, env, metrics, rewards)
            if metrics
            else None
        )

        if obs_callback is not None:
            manipulated_rewards = ma_controller.update_rewards(
                obs=obs_callback(), rewards=rewards.copy(), metrics=current_metrics
            )  # For envs where all agents see the same global observation
        else:
            manipulated_rewards = ma_controller.update_rewards(
                obs=observations, rewards=rewards.copy(), metrics=current_metrics
            )

        for agent_id, reward in rewards.items():
            episode_reward[agent_id] += reward

        controller.step_agent_parallel(
            observations, actions, manipulated_rewards, terminations, infos
        )
        controller.step_finished(timestep, new_observations)

        observations = new_observations

    env.close()
    controller.episode_finished(current_episode, "train")
    ma_controller.episode_finished(current_episode, "train")
    return episode_reward


def _eval_parallel_agents(
    controller: IController,
    env: ParallelEnv,
    writer: SummaryWriter,
    current_epoch: int,
    obs_logger: IObsLogger,
    num_eval_episodes: int,
) -> None:
    rewards: dict[AgentID, list[float]] = {}

    actions: dict[AgentID, list[ActionType]] = defaultdict(list)

    for agent_name in env.possible_agents:
        rewards[agent_name] = []
    obs_logger.clear_buffer()

    # For coin game this resets the history.Important in case e.g. eval is done in between
    env.reset(options={"history_reset": True})

    for steps in range(num_eval_episodes):
        observations, infos = env.reset(
            options={
                "write_log": False,
                "epoch": current_epoch,
                "tag": "eval",
            }
        )

        episode_reward = {}
        for agent_name in env.possible_agents:
            episode_reward[agent_name] = 0

        while env.agents:
            if get_cfg().get_render_mode() != "":
                pygame.event.get()  # so that the window doesn't freeze

            a: dict[AgentID, ActionType] = controller.act_parallel(
                observations, explore=False
            )

            for agent_id, observation in observations.items():
                obs_logger.add_observation(agent_id, observation)

            new_observations, r, terminations, truncations, infos = env.step(a)

            observations = new_observations

            for agent_id, action in a.items():
                actions[agent_id].append(action)

            for agent_id, reward in r.items():
                episode_reward[agent_id] += reward

        env.close()

        for agent_id in env.possible_agents:
            writer.add_scalar(
                f"eval/rewards/epoch-{current_epoch}/{agent_id}",
                episode_reward[agent_id],
                steps,
            )
            rewards[agent_id].append(episode_reward[agent_id])

    obs_logger.log_epoch(current_epoch, "eval")

    for agent_id in env.possible_agents:
        writer.add_scalar(
            f"eval/rewards/mean/{agent_id}",
            numpy.mean(rewards[agent_id]),
            current_epoch,
        )
        action = numpy.array(actions[agent_id])

        writer.add_histogram(
            f"eval-actions/{agent_id}", action, global_step=current_epoch
        )

    log_efficiency(current_epoch, "eval", writer, rewards, num_eval_episodes)

    # by resetting some envs (e.g. coin game) will log some env specific data
    env.reset(
        options={
            "write_log": True,
            "epoch": current_epoch,
            "tag": "eval",
            "heatmap": True,
        }
    )


def log_configs(writer: SummaryWriter) -> None:
    for key, value in _training_config.__dict__.items():
        writer.add_text(key, str(value))


def start_training() -> None:
    logging.info(f"Starting training")

    agent_dir: str = get_cfg().get_ctrl_dir()

    logging.info(f"TensorBoard -> Logging to {agent_dir}")

    with SummaryWriter(log_dir=agent_dir) as writer:
        # Building  the environment
        env: ParallelEnv | AECEnv = build_env(get_cfg().exp_config.ENV_NAME, writer)

        parallel: bool = isinstance(env, ParallelEnv)
        if not parallel:
            raise NotImplementedError("Only parallel envs are supported")

        obs_logger: IObsLogger

        if isinstance(env.unwrapped, SimpleEnv):
            logging.info("Using SimpleEnvLogger")
            obs_logger = SimpleEnvLogger(env.unwrapped, writer)  # type: ignore
        else:
            logging.info("Using IObsLogger")
            obs_logger = IObsLogger(writer)

        # Building the agents
        agents = get_agents()

        agents.init_agents(
            action_space={a: env.action_space(a) for a in env.possible_agents},
            observation_space={
                a: env.observation_space(a) for a in env.possible_agents
            },
        )

        ma_controller: IMaController = get_ma_controller(
            _training_config.MA_MODE, get_cfg().exp_config.ENV_NAME
        )
        ma_controller.set_logger(writer)
        ma_controller.set_agents(
            env.possible_agents, get_obs_space(_training_config.MA_MODE, env)
        )

        agents.set_logger(writer)

        log_configs(writer)
        if get_cfg().get_render_mode() != "":
            pygame.init()

        # if fast first results are needed
        # for epoch in range(1, _training_config.EPOCHS + 1):
        # normal evaluation/epoch counter
        for epoch in range(0, _training_config.EPOCHS + 1):
            _train_epoch(
                ma_controller, agents, env, epoch, writer, obs_logger, parallel=parallel
            )

            if epoch % _training_config.EVAL_EPOCH_INTERVAL == 0:
                logging.info("Evaluating agents")
                if parallel:
                    _eval_parallel_agents(
                        agents,
                        env,
                        writer,
                        epoch,
                        obs_logger,
                        num_eval_episodes=_training_config.EVAL_EPISODES,
                    )
                else:
                    raise NotImplementedError("Only parallel envs are supported")

                # save model
                logging.info("Saving model")
                agents.save(get_cfg().get_model_storage(epoch))

        env.close()
        logging.info("Finished training")


def start() -> None:
    set_cfg(CfgManager(_training_config))
    logging.basicConfig(level=logging.getLevelName(get_cfg().get_log_level()))
    logging.info("Starting MR-in-MARL")
    start_training()


if __name__ == "__main__":
    start()
