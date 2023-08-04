import time
from datetime import datetime
from os.path import join, dirname, realpath

import numpy
from torch.utils.tensorboard import SummaryWriter
import pygame
from pettingzoo import ParallelEnv, AECEnv

import logging
from agents.agents_helper import get_agents
from agents.agents_i import IAgents
from config import (
    config,
    actor_critic_config,
    training_config,
    env_config,
    eval_config,
)
from enviroments import env_helper_parallel, env_helper_aec

logging.basicConfig(level=logging.getLevelName(config.LOG_LEVEL))


gamma = 0.95
all_steps = 0


def _episode_test(
    agents: IAgents, env: AECEnv, current_episode: int, writer: SummaryWriter
) -> dict:
    env.reset()
    agents.init_new_episode()

    episode_reward = {}
    last_action = {}
    last_observation = {}
    last_log_prob = {}
    actions = {}

    for agent_id in env.possible_agents:
        episode_reward[agent_id] = 0
        actions[agent_id] = []

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
                log_prob=last_log_prob[agent_id],
                gamma=actor_critic_config.DISCOUNT_FACTOR,
            )

        if termination or truncation:
            action = None
            log_prob = None
        else:
            action, log_prob = agents.act(agent_id=agent_id, observation=observation)

        last_action[agent_id] = action
        last_observation[agent_id] = observation
        last_log_prob[agent_id] = log_prob
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
    for agent_id in episode_reward:
        action = numpy.array(actions[agent_id])
        writer.add_histogram(
            f"actions/{agent_id}", action, global_step=current_episode, max_bins=10
        )

    return episode_reward


def _eval_agents(
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
                action, _ = agents.act(
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


def get_logging_file(run_name: str) -> str:
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
    baseline_values = {"agent_0": -39.79}
    log_location = get_logging_file("baseline/simple")

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


def start_training() -> None:
    env: ParallelEnv | AECEnv = (
        env_helper_parallel.setup_env()
        if env_config.PARALLEL_ENV
        else env_helper_aec.setup_env()
    )
    RUN_NAME = f"{env_config.ENV_NAME}/{training_config.AGENT_TYPE.value}/{datetime.fromtimestamp(time.time()).isoformat(timespec='seconds')} - {config.NAME_TAG}"
    ml_folder = get_logging_file(RUN_NAME)
    logging.info(f"TensorBoard -> Logging to {ml_folder}")

    agents = get_agents()
    logging.info(f"Starting agent type: {training_config.AGENT_TYPE}")

    agents.init_agents(
        action_space={a: env.action_space(a) for a in env.possible_agents},
        observation_space={a: env.observation_space(a) for a in env.possible_agents},
    )

    with SummaryWriter(log_dir=ml_folder) as writer:
        agents.set_logger(writer)

        for key, value in actor_critic_config.__dict__.items():
            writer.add_text(key, str(value))
        for key, value in training_config.__dict__.items():
            writer.add_text(key, str(value))
        for key, value in env_config.__dict__.items():
            writer.add_text(key, str(value))

        pygame.init()
        all_rewards = {}

        for epoch in range(training_config.EPOCHS):
            for episode in range(training_config.EPISODES):
                if env_config.PARALLEL_ENV:
                    raise NotImplementedError
                else:
                    if episode % eval_config.EVAL_INTERVAL == 0:
                        logging.info("Evaluating agents")
                        _eval_agents(
                            agents,
                            env,
                            writer,
                            episode,
                            num_eval_episodes=eval_config.NUM_EPISODES,
                        )
                        # save model

                        model_storage = join(
                            dirname(dirname(realpath(__file__))),
                            "resources",
                            "models",
                            RUN_NAME,
                            f"episode-{episode}",
                        )

                        agents.save(model_storage)
                    rewards = _episode_test(agents, env, episode, writer)

                    for i in rewards:
                        if i not in all_rewards:
                            all_rewards[i] = []
                        all_rewards[i].append(rewards[i])
                # for agent_name in rewards:
                #     writer.log_metric(
                #         key=agent_name,
                #         value=rewards[agent_name],
                #         step=episode * (1 + epoch),
                #     )
                logging.info(
                    f"Epoch {epoch}/{training_config.EPOCHS} Episode: {episode}/{training_config.EPISODES} - Reward: {rewards}"
                )

        env.close()
        logging.info("Finished training")

        #
        for agent_id in env.possible_agents:
            logging.info(f"{agent_id} mean reward: {numpy.mean(all_rewards[agent_id])}")


if __name__ == "__main__":
    logging.info("Starting MR-in-MARL")
    start_training()
    # baseline_writer()
