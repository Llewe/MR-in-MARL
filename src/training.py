import logging
from collections import defaultdict

import numpy
import pygame
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from pettingzoo.utils.env import AgentID
from torch.utils.tensorboard import SummaryWriter

from src.agents.utils.agents_helper import get_agents
from src.cfg_manager import CfgManager, get_cfg, set_cfg
from src.config.training_config import TrainingConfig
from src.envs import build_env
from src.interfaces.agents_i import IAgents
from src.utils.loggers.obs_logger import ObsLogger
from src.utils.loggers.simple_env_logger import SimpleEnvLogger

_training_config = TrainingConfig()


all_steps = 0


def _train_aec_epoch(
    agents: IAgents,
    env: AECEnv,
    current_epoch: int,
    writer: SummaryWriter,
    obs_logger: ObsLogger,
) -> None:
    agents.epoch_started(current_epoch)

    # For coin game this resets the history.Important in case e.g. eval is done in between
    env.reset(options={"history_reset": True})

    for episode in range(1, _training_config.EPISODES + 1):
        logging.info(
            f"Epoch {current_epoch}/{_training_config.EPOCHS}"
            f" Episode: {episode}/{_training_config.EPISODES}"
        )

        _train_aec_episode_simple(
            agents, env, current_epoch, episode, writer, obs_logger
        )

    # by resetting some envs (e.g. coin game) will log some env specific data
    env.reset(
        options={
            "write_log": True,
            "epoch": current_epoch,
            "tag": "train",
        }
    )

    agents.epoch_finished(current_epoch)


def _train_aec_episode_simple(
    agents: IAgents,
    env: AECEnv,
    current_epoch: int,
    current_episode: int,
    writer: SummaryWriter,
    obs_logger: ObsLogger,
) -> None:
    env.reset(
        options={
            "write_log": False,
            "epoch": current_epoch,
            "tag": "train",
        }
    )

    episode_reward: dict[AgentID, float] = defaultdict(lambda: 0)
    actions = defaultdict(list)
    # last_observation = {}

    obs_logger.clear_buffer()
    for agent_id in env.agent_iter():
        pygame.event.get()  # so that the window doesn't freeze
        observation, reward, termination, truncation, info = env.last()

        obs_logger.add_observation(agent_id, observation)

        action = agents.act(agent_id=agent_id, observation=observation)
        agents.update(
            agent_id,
            observation,
            action,
            reward,
            termination or truncation,
        )

        if termination or truncation:
            env.step(None)
            continue

        env.step(action)

        # logging
        episode_reward[agent_id] += reward

        if action is not None:
            actions[agent_id].append(action)
            # last_observation[agent_id]= observation

    obs_logger.log_episode(current_episode, "train")

    for agent_id in episode_reward:
        writer.add_scalar(
            f"rewards/{agent_id}", episode_reward[agent_id], current_episode
        )
        action = numpy.array(actions[agent_id])

        writer.add_histogram(f"actions/{agent_id}", action, global_step=current_episode)


def _eval_aec_agents(
    agents: IAgents,
    env: AECEnv,
    writer: SummaryWriter,
    current_epoch: int,
    obs_logger: ObsLogger,
    num_eval_episodes: int,
) -> None:
    rewards: dict[AgentID, list[float]] = {}

    for agent_name in env.possible_agents:
        rewards[agent_name] = []
    obs_logger.clear_buffer()

    # For coin game this resets the history.Important in case e.g. eval is done in between
    env.reset(options={"history_reset": True})

    for steps in range(num_eval_episodes):
        env.reset(
            options={
                "write_log": False,
                "epoch": current_epoch,
                "tag": "eval",
            }
        )

        episode_reward = {}
        for agent_name in env.possible_agents:
            episode_reward[agent_name] = 0

        for agent_name in env.agent_iter():
            pygame.event.get()  # so that the window doesn't freeze
            observation, reward, termination, truncation, info = env.last()

            obs_logger.add_observation(agent_name, observation)

            episode_reward[agent_name] += reward

            if termination or truncation:
                action = None
            else:
                action = agents.act(
                    agent_id=agent_name, observation=observation, explore=False
                )

            env.step(action)

        for agent_id in env.possible_agents:
            writer.add_scalar(
                f"eval/rewards/epoch-{current_epoch}/{agent_id}",
                episode_reward[agent_id],
                steps,
            )
            rewards[agent_id].append(episode_reward[agent_id])

    obs_logger.log_episode(current_epoch, "eval")

    for agent_id in env.possible_agents:
        writer.add_scalar(
            f"eval/rewards/mean/{agent_id}",
            numpy.mean(rewards[agent_id]),
            current_epoch,
        )

    # by resetting some envs (e.g. coin game) will log some env specific data
    env.reset(
        options={
            "write_log": True,
            "epoch": current_epoch,
            "tag": "eval",
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

        obs_logger: ObsLogger

        if isinstance(env.unwrapped, SimpleEnv):
            logging.info("Using SimpleEnvLogger")
            obs_logger = SimpleEnvLogger(env.unwrapped, writer)  # type: ignore
        else:
            logging.info("Using ObsLogger")
            obs_logger = ObsLogger(writer)

        # Building the agents
        agents = get_agents()

        agents.init_agents(
            action_space={a: env.action_space(a) for a in env.possible_agents},
            observation_space={
                a: env.observation_space(a) for a in env.possible_agents
            },
        )

        agents.set_logger(writer)

        log_configs(writer)

        pygame.init()

        for epoch in range(1, _training_config.EPOCHS + 1):
            _train_aec_epoch(agents, env, epoch, writer, obs_logger)

            if epoch % _training_config.EVAL_EPOCH_INTERVAL == 0:
                logging.info("Evaluating agents")
                _eval_aec_agents(
                    agents,
                    env,
                    writer,
                    epoch,
                    obs_logger,
                    num_eval_episodes=_training_config.EVAL_EPISODES,
                )

                # save model
                logging.info("Saving model")
                agents.save(get_cfg().get_model_storage(epoch))

        env.close()
        logging.info("Finished training")


if __name__ == "__main__":
    set_cfg(CfgManager(_training_config))
    logging.basicConfig(level=logging.getLevelName(get_cfg().get_log_level()))
    logging.info("Starting MR-in-MARL")
    start_training()
