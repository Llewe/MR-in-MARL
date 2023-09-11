from time import time
from collections import defaultdict
from datetime import datetime
from os.path import join, dirname, realpath

import numpy
from torch.utils.tensorboard import SummaryWriter
import pygame
from pettingzoo import ParallelEnv, AECEnv

import logging
from agents.agents_helper import get_agents
from src.interfaces.agents_i import IAgents
from src import (
    log_config,
    actor_critic_config,
    training_config,
    env_config,
    eval_config,
)
from enviroments import build_env

logging.basicConfig(level=logging.getLevelName(log_config.LOG_LEVEL))


gamma = 0.95
all_steps = 0


def _train_aec_episode(
    agents: IAgents, env: AECEnv, current_episode: int, writer: SummaryWriter
) -> dict:
    env.reset()
    agents.init_new_episode()

    episode_reward = defaultdict(lambda: 0)
    last_action = {}
    last_observation = {}
    actions = defaultdict(list)

    for agent_id in env.agent_iter():
        pygame.event.get()  # so that the window doesn't freeze
        observation, reward, termination, truncation, info = env.last()

        # lets update the agent from the last cycle, if there was one
        if agent_id in last_action:
            agents.update(
                agent_id=agent_id,
                last_observation=last_observation[agent_id],
                curr_observation=observation,
                last_action=last_action[agent_id],
                reward=reward,
                done=termination or truncation,
                gamma=actor_critic_config.DISCOUNT_FACTOR,
            )
        if termination or truncation:
            action = None
        else:
            action = agents.act(agent_id=agent_id, observation=observation)

        last_action[agent_id] = action
        last_observation[agent_id] = observation
        global all_steps
        all_steps += 1
        writer.add_scalar(f"all_rewards", reward, all_steps)
        episode_reward[agent_id] += reward

        if action is not None:
            actions[agent_id].append(action)

        env.step(action)

    for agent_id in episode_reward:
        writer.add_scalar(
            f"rewards/{agent_id}", episode_reward[agent_id], current_episode
        )
        action = numpy.array(actions[agent_id])
        writer.add_histogram(
            f"actions/{agent_id}", action, global_step=current_episode, max_bins=10
        )

    return episode_reward


def _eval_aec_agents(
    agents: IAgents,
    env: AECEnv,
    writer: SummaryWriter,
    current_episode: int,
    num_eval_episodes: int,
):
    rewards = {}
    for agent_name in env.possible_agents:
        rewards[agent_name] = []

    for steps in range(num_eval_episodes):
        env.reset()

        episode_reward = {}
        for agent_name in env.possible_agents:
            episode_reward[agent_name] = 0

        for agent_name in env.agent_iter():
            pygame.event.get()  # so that the window doesn't freeze
            observation, reward, termination, truncation, info = env.last()

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
                f"eval/rewards/episode-{current_episode}/{agent_id}",
                episode_reward[agent_id],
                steps,
            )
            rewards[agent_id].append(episode_reward[agent_id])

    for agent_id in env.possible_agents:
        writer.add_scalar(
            f"eval/rewards/mean/{agent_id}",
            numpy.mean(rewards[agent_id]),
            current_episode,
        )


def get_logging_folder(run_name: str) -> str:
    file = join(
        dirname(dirname(realpath(__file__))),
        "resources",
        "tensorboard",
        run_name,
    )
    return file


def baseline_writer():
    steps = 10000
    # simple:
    # baseline_values = {"agent_0": -39.79}
    # log_location = get_logging_file("baseline/simple")
    # simple_adversary:
    baseline_values = {
        "adversary_0": -28.19,
        "agent_0": 7.40,
        "agent_1": 7.40,
    }
    log_location = get_logging_folder("baseline/simple_adversary")
    # baseline_values = {
    #     "adversary_0": 4.51,
    #     "adversary_1": 4.51,
    #     "adversary_2": 4.51,
    #     "agent_0": -16.25,
    # }
    # log_location = get_logging_file("baseline/simple_tag")

    log_name = "rewards"

    with SummaryWriter(log_dir=log_location) as writer:
        for i in range(steps):
            for agent_name in baseline_values:
                writer.add_scalar(
                    f"{log_name}/{agent_name}", baseline_values[agent_name], i
                )


def create_run_name() -> str:
    env_name: str = env_config.ENV_NAME
    agent_name: str = training_config.AGENT_TYPE.value
    cur_time: str = datetime.fromtimestamp(time()).isoformat(timespec="seconds")
    tag: str = log_config.NAME_TAG
    return f"{env_name}/{agent_name}/{cur_time} - {tag}"


def get_model_storage(name: str, episode: int) -> str:
    return join(
        dirname(dirname(realpath(__file__))),
        "resources",
        "models",
        name,
        f"episode-{episode + 1}",
    )


def log_configs(writer: SummaryWriter) -> None:
    for key, value in actor_critic_config.__dict__.items():
        writer.add_text(key, str(value))
    for key, value in training_config.__dict__.items():
        writer.add_text(key, str(value))
    for key, value in env_config.__dict__.items():
        writer.add_text(key, str(value))


def start_training() -> None:
    logging.info(f"Starting training")

    run_name: str = create_run_name()
    log_folder: str = get_logging_folder(run_name)

    logging.info(f"TensorBoard -> Logging to {log_folder}")

    # Building  the environment
    env: ParallelEnv | AECEnv = build_env(env_config.ENV_NAME)

    # Building the agents
    agents = get_agents()

    agents.init_agents(
        action_space={a: env.action_space(a) for a in env.possible_agents},
        observation_space={a: env.observation_space(a) for a in env.possible_agents},
    )

    with SummaryWriter(log_dir=log_folder) as writer:
        agents.set_logger(writer)

        log_configs(writer)

        pygame.init()

        for epoch in range(training_config.EPOCHS):
            for episode in range(training_config.EPISODES):
                logging.info(
                    f"Epoch {epoch}/{training_config.EPOCHS}"
                    f" Episode: {episode}/{training_config.EPISODES}"
                )

                _train_aec_episode(agents, env, episode, writer)

                if (episode + 1) % eval_config.EVAL_INTERVAL == 0:
                    logging.info("Evaluating agents")
                    _eval_aec_agents(
                        agents,
                        env,
                        writer,
                        episode + 1,
                        num_eval_episodes=eval_config.NUM_EPISODES,
                    )

                    # save model
                    logging.info("Saving model")
                    agents.save(get_model_storage(run_name, episode + 1))

        env.close()
        logging.info("Finished training")


if __name__ == "__main__":
    logging.info("Starting MR-in-MARL")
    start_training()
    # baseline_writer()
