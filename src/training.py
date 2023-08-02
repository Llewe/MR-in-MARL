import time
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
    gamma = 0.95

    for agent_name in env.possible_agents:
        episode_reward[agent_name] = 0
        actions[agent_name] = []

    for agent_name in env.agent_iter():
        pygame.event.get()  # so that the window doesn't freeze
        observation, reward, termination, truncation, info = env.last()

        # lets update the agent from the last cycle, if there was one
        if agent_name in last_action:
            agents.update(
                agent_id=agent_name,
                last_observation=last_observation[agent_name],
                curr_observation=observation,
                last_action=last_action[agent_name],
                reward=reward,
                done=termination or truncation,
                log_prob=last_log_prob[agent_name],
                gamma=gamma,
            )

        if termination or truncation:
            action = None
            log_prob = None
        else:
            # if numpy.random.randint(current_episode + 1) < 50:
            #     print("random action")
            #     action = numpy.random.choice(env.action_space(agent_name).n)
            #     log_prob = 1.0
            # else:
            action, log_prob = agents.act(agent_id=agent_name, observation=observation)

        last_action[agent_name] = action
        last_observation[agent_name] = observation
        last_log_prob[agent_name] = log_prob
        global all_steps
        all_steps += 1
        writer.add_scalar(f"all_rewards", reward, all_steps)
        episode_reward[agent_name] += reward

        if action is not None:
            actions[agent_name].append(action)

        env.step(action)

    for agent_name in episode_reward:
        writer.add_scalar(
            f"rewards/{agent_name}", episode_reward[agent_name], current_episode
        )
    for agent_name in episode_reward:
        action = numpy.array(actions[agent_name])
        writer.add_histogram(
            f"actions/{agent_name}", action, global_step=current_episode, max_bins=10
        )

    return episode_reward


# def _episode_parallel(
#     actor_critic: ActorCritic, env: ParallelEnv, current_episode: int
# ) -> dict:
#     observations, states = env.reset()
#
#     episode_reward = {}
#     for agent_name in observations:
#         episode_reward[agent_name] = 0
#
#     steps = 0
#     global gamma
#     gamma -= 0.001
#     if gamma < 0.2:
#         gamma = 0.2
#     mlflow.log_metric("gamma", gamma, current_episode)
#     while env.agents:
#         pygame.event.get()  # so that the window doesn't freeze
#         steps += 1
#
#         actions, log_prob = actor_critic.act(observations)
#
#         next_observations, rewards, terminations, truncations, infos = env.step(actions)
#
#         actor_critic.update(
#             observations,
#             actions,
#             rewards,
#             next_observations,
#             terminations,
#             log_prob,
#             gamma,
#         )
#         observations = next_observations
#
#         for agent_name in rewards:
#             episode_reward[agent_name] += rewards[agent_name]
#
#     return episode_reward
#
#
# def _episode_aec(actor_critic: ActorCritic, env: AECEnv, current_episode: int) -> dict:
#     env.reset()
#
#     episode_reward = {}
#     last_action = {}
#     last_observation = {}
#     last_log_prob = {}
#     for agent_name in env.possible_agents:
#         episode_reward[agent_name] = 0
#     for agent_name in env.agent_iter():
#         pygame.event.get()  # so that the window doesn't freeze
#
#         observation, reward, termination, truncation, info = env.last()
#
#         if agent_name in last_action:
#             actor_critic.update_single(
#                 agent_name,
#                 last_observation[agent_name],
#                 last_action[agent_name],
#                 reward,
#                 observation,
#                 termination or truncation,
#                 last_log_prob[agent_name],
#                 0.95,
#             )
#
#         if env.agent_selection is not agent_name:
#             raise Exception
#
#         if termination or truncation:
#             action = None
#
#         else:
#             action, log_prob = actor_critic.act_single(agent_name, observation)
#             last_log_prob[agent_name] = log_prob
#
#         last_action[agent_name] = action
#         last_observation[agent_name] = observation
#
#         env.step(action)
#
#         episode_reward[agent_name] += reward
#
#     return episode_reward


def start_training() -> None:
    env: ParallelEnv | AECEnv = (
        env_helper_parallel.setup_env()
        if env_config.PARALLEL_ENV
        else env_helper_aec.setup_env()
    )

    RUN_NAME = f"a-{training_config.AGENT_TYPE.value}_e-{env_config.ENV_NAME}/time-{time.time()}"
    ml_folder = join(
        dirname(dirname(realpath(__file__))),
        "resources",
        "tensorboard",
        RUN_NAME,
    )
    logging.info(f"TensorBoard -> Logging to {ml_folder}")

    agents = get_agents()
    logging.info(f"Starting agent type: {training_config.AGENT_TYPE}")

    agents.init_agents(
        action_space={a: env.action_space(a) for a in env.possible_agents},
        observation_space={a: env.observation_space(a) for a in env.possible_agents},
    )

    with SummaryWriter(log_dir=ml_folder) as writer:
        agents.set_logger(writer)
        # writer.add_hparams(hparam_dict=vars(actor_critic_config), metric_dict={})
        # writer.add_hparams(hparam_dict=vars(actor_critic_config), metric_dict={})
        # writer.add_hparams(hparam_dict=vars(training_config), metric_dict={})
        # writer.add_hparams(hparam_dict=vars(env_config), metric_dict={})
        pygame.init()
        for epoch in range(training_config.EPOCHS):
            for episode in range(training_config.EPISODES):
                if env_config.PARALLEL_ENV:
                    raise NotImplementedError
                else:
                    rewards = _episode_test(agents, env, episode, writer)

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


if __name__ == "__main__":
    logging.info("Starting MR-in-MARL")
    start_training()
