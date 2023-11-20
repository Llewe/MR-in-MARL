# reference
# https://github.com/arjun-prakash/pz_dilemma
# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/rps/rps.py
# https://github.com/tianyu-z/pettingzoo_dilemma_envs
import math
import random

import gymnasium
import numpy as np
import pygame
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from torch.utils.tensorboard import SummaryWriter

SEED = 42
from copy import deepcopy


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv):
    """Two-player environment for rock paper scissors.
    Expandable environment to rock paper scissors lizard spock action_6 action_7 ...
    The observation is simply the last opponent action.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "coin_game_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        num_actions=4,
        max_cycles=15,
        render_mode=None,
        nb_players=2,
        grid_size=3,
        randomize_coin=False,
        allow_overlap_players=False,
        summary_writer: SummaryWriter = None,
    ):
        pygame.init()
        self.renderOn = False
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])

        self.nb_players = nb_players
        self.max_cycles = max_cycles
        self.render_mode = "human"
        self.name = "coin_game_v0"
        self.randomize_coin = randomize_coin
        self.allow_overlap_players = allow_overlap_players
        self.summary_writer = summary_writer

        self.current_reset_counter = 0
        self.current_reset_eval_counter = 0
        self._moves = [
            np.array([0, 1]),  # right
            np.array([0, -1]),  # left
            np.array([1, 0]),  # down
            np.array([-1, 0]),  # up
        ]
        # none is last possible action, to satisfy discrete action space
        self._none = np.array([0, 0])

        self.agents = ["player_" + str(r) for r in range(self.nb_players)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.nb_players))))
        self.action_spaces = {agent: Discrete(num_actions) for agent in self.agents}

        self.grid_size = grid_size
        self.ob_space_shape = [
            self.nb_players * 2,
            self.grid_size,
            self.grid_size,
        ]  # *2 because we need the pos of the coin and the player itself, they are paired
        self.grids = [
            [i, j] for i in range(self.grid_size) for j in range(self.grid_size)
        ]

        self.state = {
            agent: {"px": -1, "py": -1, "cx": -1, "cy": -1, "hc": 0}
            for agent in self.agents
        }  # init the location of agents and the coin player_x, player_y, coin_x, coin_y, has_coin
        self.observation_spaces = {
            agent: Discrete(
                (len(self.state["player_0"].values()) + 1 + 1) * self.nb_players
            )
            for agent in self.agents
        }
        self.observations = {
            agent: [0]
            * (len(self.state["player_0"].values()) + 1 + 1)
            * self.nb_players
            for agent in self.agents
        }
        self.coin_pos = -1 * np.ones(2, dtype=np.int8)
        self.coin_pos_old = deepcopy(self.coin_pos)
        self.player_pos = -1 * np.ones(
            (self.nb_players, 2)
        )  # the x-coord and y-coord of the player
        self.player_coin = np.random.randint(
            self.nb_players
        )  # the generated coin's color
        self.state[self.agents[self.player_coin]][
            "hc"
        ] = 1  # the player who has the coin
        self.render_mode = render_mode

        # aux variables during steps
        self.actions_taken = {agent: None for agent in self.agents}
        self.generate_new_coin = False
        if randomize_coin:
            self.agent_picker_buffer = np.array(range(self.nb_players))
            random.shuffle(self.agent_picker_buffer)
            self.agent_picker_idx = 0

        self.agents_collected_coins: dict[str:int] = {agent: 0 for agent in self.agents}
        self.steps_to_collect: dict[str:int] = {agent: 0 for agent in self.agents}
        self.reinit()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _generate_coin(self, randomize=False):
        self.player_coin_old = self.player_coin

        if randomize:
            # next coin belong to a random agent
            self.player_coin = self.agent_picker_buffer[
                self.agent_picker_idx % self.nb_players
            ]
            self.agent_picker_idx += 1
            if self.agent_picker_idx % self.nb_players == 0:
                random.shuffle(self.agent_picker_buffer)
        else:
            self.player_coin = (
                1 + self.player_coin
            ) % self.nb_players  # next coin belong to next agent
        self.grids_copy = deepcopy(self.grids)
        for j in range(self.nb_players):
            if (
                list(self.player_pos[j, :]) in self.grids_copy
            ):  # TODO commit this change to the original repo
                self.grids_copy.remove(list(self.player_pos[j, :]))
        random.shuffle(self.grids_copy)
        self.coin_pos = np.array(self.grids_copy[0])
        return

    def _generate_state(self):
        for j in range(self.nb_players):
            self.state[self.agents[j]]["px"] = self.player_pos[j][0]
            self.state[self.agents[j]]["py"] = self.player_pos[j][1]
            if self.player_coin == j:
                self.state[self.agents[j]]["cx"] = self.coin_pos[0]
                self.state[self.agents[j]]["cy"] = self.coin_pos[1]
                self.state[self.agents[j]]["hc"] = 1
        return

    def _generate_observation(self):
        state_and_action_and_reward = deepcopy(self.state)
        for a in self.agents:
            state_and_action_and_reward[a]["action"] = self.actions_taken[a]
            state_and_action_and_reward[a]["reward"] = self.rewards[a]
        tmp = []
        for a in self.agents:
            tmp += list(state_and_action_and_reward[a].values())
        for a in self.agents:
            self.observations[a] = deepcopy(tmp)
        del state_and_action_and_reward
        return

    def is_reset_stats(self, options=None) -> bool:
        if options and "eval" in options:
            return options["eval"]
        return True

    def reinit(self, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.player_pos = -1 * np.ones((self.nb_players, 2))
        self.coin_pos = -1 * np.ones(2, dtype=np.int8)
        # self.state = {agent: self._none for agent in self.agents}

        # generate player_pos
        self.grids_copy = deepcopy(self.grids)
        random.shuffle(self.grids_copy)
        self.player_pos[:, :] = np.array(self.grids_copy[: self.nb_players])
        # generate coin pos
        self._generate_coin(self.randomize_coin)

        # aux variables during steps
        self.generate_new_coin = False
        self.actions_taken = {agent: None for agent in self.agents}

        self._generate_state()
        self.observations = {
            agent: [0]
            * (len(self.state["player_0"].values()) + 1 + 1)
            * self.nb_players
            for agent in self.agents
        }
        self.num_moves = 0

        if self.is_reset_stats(options):
            if self.summary_writer:
                self.log_episode_metrics(
                    self.current_reset_counter, eval=options and "eval" in options
                )

            self.agents_collected_coins: dict[str:int] = {
                agent: 0 for agent in self.agents
            }
            self.steps_to_collect: dict[str:int] = {agent: 0 for agent in self.agents}
            self.current_reset_eval_counter = 0
            if not options or "eval" not in options:
                self.current_reset_counter += 1
        else:
            self.current_reset_eval_counter += 1

    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        if self.render_mode == "text":
            print("This is a {} round".format(self.num_moves))
            print("Coin (before taken) position: {}".format(self.coin_pos))
            print("Coin (before taken) belongs to: {}".format(self.player_coin))
            agent = self.agent_selection
            if len(self.agents) == self.nb_players:
                # print("Players information: ")
                print(
                    "Agent {} position before action: {} ".format(
                        agent,
                        self.player_pos_old[self.agent_name_mapping[agent], :],
                    )
                )
                print(
                    "Agent {} action: {} ".format(
                        agent, self._moves[self.actions_taken[agent]]
                    )
                )
                print(
                    "Agent {} position after action: {} ".format(
                        agent, self.player_pos[self.agent_name_mapping[agent], :]
                    )
                )
                if self._agent_selector.is_last():
                    for a in self.agents:
                        print(
                            "Agent {} reward after action: {} ".format(
                                a, self.rewards[a]
                            )
                        )
                        print(
                            "Agent {} cumulative rewards after action: {} ".format(
                                a, self._cumulative_rewards[a]
                            )
                        )
            else:
                print("Game over")
            print("\n")
        elif self.render_mode == "human":
            self.enable_render(self.render_mode)
            self.draw()
            pygame.display.flip()

    def draw(self):
        colour_background = (48, 48, 48)
        color_grid = (255, 255, 255)
        color_players = [
            (233, 30, 99),
            (156, 39, 176),
            (103, 58, 183),
            (63, 81, 181),
            (33, 150, 243),
            (0, 188, 212),
            (0, 150, 136),
        ]
        color_coin = (255, 0, 0)

        # clear screen
        self.screen.fill(colour_background)

        # draw grid
        cell_size_w = self.width / self.grid_size
        cell_size_h = self.height / self.grid_size

        player_size = (
            0.8 * cell_size_w / 2
            if cell_size_w < cell_size_h
            else 0.8 * cell_size_h / 2
        )
        size_coin = (
            0.2 * cell_size_w / 2
            if cell_size_w < cell_size_h
            else 0.2 * cell_size_h / 2
        )

        for w in range(1, self.grid_size):
            pos_w = w * cell_size_w

            for h in range(1, self.grid_size):
                pos_h = h * cell_size_h
                pygame.draw.line(
                    self.screen,
                    color_grid,
                    (pos_w, 0),
                    (pos_w, self.height),
                )
                pygame.draw.line(
                    self.screen,
                    color_grid,
                    (0, pos_h),
                    (self.width, pos_h),
                )
        player_degree = 2 * math.pi / self.nb_players
        # draw players
        for i, pos in enumerate(self.player_pos):
            p_x = pos[0] * cell_size_w + cell_size_w / 2
            p_y = pos[1] * cell_size_h + cell_size_h / 2

            rec = pygame.Rect(
                p_x - player_size, p_y - player_size, player_size * 2, player_size * 2
            )

            player_degree_start = i * player_degree
            player_degree_end = (i + 1) * player_degree
            pygame.draw.arc(
                self.screen,
                color_players[i],
                rec,
                start_angle=player_degree_start,
                stop_angle=player_degree_end,
                width=25,
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet

        # draw coins
        c_x = self.coin_pos[0] * cell_size_w + cell_size_w / 2
        c_y = self.coin_pos[1] * cell_size_h + cell_size_h / 2

        rec = pygame.Rect(
            c_x - size_coin, c_y - size_coin, size_coin * 2, size_coin * 2
        )

        pygame.draw.arc(
            self.screen,
            color_players[self.player_coin],
            rec,
            start_angle=0,
            stop_angle=360,
            width=10,
        )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet

        # update bounds to center around agent

    def observe(self, agent):
        # observation of one agent is the previous state of the other
        print("state")
        print(self.state)
        print("player_0")
        print(self.observations["player_0"])
        print("player_1")
        print(self.observations["player_1"])
        print("player_2")
        print(self.observations["player_2"])
        print("player_3")
        print(self.observations["player_3"])
        print(
            "it looks like this env is buggy -> at some point everyone is coin holder"
        )

        return np.array(self.observations[agent])

    def close(self):
        pass

    def reset(self, seed=SEED, return_info=False, options=None):
        self.reinit(options=options)

    def _same_pos(self, x, y):
        return (x == y).all()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        agent = self.agent_selection
        self.steps_to_collect[agent] += 1

        if self._agent_selector.is_first():
            self.rewards = {agent: 0 for agent in self.agents}
        # self.state[self.agent_selection] = action
        self.generate_new_coin = False

        self.actions_taken[agent] = action
        if not self.allow_overlap_players:
            self.player_pos_old = deepcopy(self.player_pos)
            potential_position = (
                self.player_pos[self.agent_name_mapping[agent], :] + self._moves[action]
            ) % self.grid_size
            if potential_position.tolist() not in self.player_pos.tolist():
                self.player_pos[self.agent_name_mapping[agent], :] = potential_position
            # if this grid already has an agent, stay still
        else:
            self.player_pos[self.agent_name_mapping[agent], :] = (
                self.player_pos[self.agent_name_mapping[agent], :] + self._moves[action]
            ) % self.grid_size

        # when all agent is done, update state and generate observations
        if self._agent_selector.is_last():
            self.rewards = {agent: 0 for agent in self.agents}
            for a in self.agents:
                # compute rewards
                if self.player_coin == self.agent_name_mapping[a]:
                    if self._same_pos(
                        self.player_pos[self.agent_name_mapping[a]],
                        self.coin_pos,
                    ):
                        self.generate_new_coin = True
                        self.rewards[a] += 1

                        self.agents_collected_coins[a] += 1

                    for k in range(self.nb_players):
                        if k != self.agent_name_mapping[a]:
                            if self._same_pos(self.player_pos[k], self.coin_pos):
                                self.generate_new_coin = True
                                self.rewards[a] -= 2
                                self.rewards["player_" + str(k)] += 1
            # if a coin is collected and all agents finish their actions, regenerate the coin

        if self._agent_selector.is_last():
            self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        if self._agent_selector.is_last():
            self.coin_pos_old = deepcopy(self.coin_pos)
            if self.generate_new_coin:
                self._generate_coin((self.randomize_coin))
            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_cycles for agent in self.agents
            }
            self._generate_state()
            # observe the current states
            self._generate_observation()  # each agent knows the all the state, action and reward of all the agents

        # self._cumulative_rewards[self.agent_selection] = 0

        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def log_episode_metrics(self, current_episode: int, eval: bool = False) -> None:
        """
        Log various episode metrics to tensorboard. This method should be called after every episode.
        Parameters
        ----------
        writer: SummaryWriter
        current_episode: int
        eval: bool
        Returns
        -------

        """
        n_coins: int = 0
        all_steps: int = 0

        log_name: str = "eval" if eval else "train"
        log_name = f"coin_game-{log_name}"

        for agent_id in self.agents:
            collected_coins = self.agents_collected_coins[agent_id]
            steps = self.steps_to_collect[agent_id]
            if eval:
                save_current_reset_eval_counter = max(
                    self.current_reset_eval_counter, 1
                )
                collected_coins = collected_coins / save_current_reset_eval_counter
                steps = steps / save_current_reset_eval_counter

            self.summary_writer.add_scalar(
                f"{log_name}/collected_coins/{agent_id}",
                collected_coins,
                current_episode,
            )
            self.summary_writer.add_scalar(
                f"{log_name}/steps_to_collect/{agent_id}",
                steps / max(collected_coins, 1),
                current_episode,
            )

            all_steps += steps
            n_coins += collected_coins
        self.summary_writer.add_scalar(
            f"{log_name}/collected_coins/all", n_coins, current_episode
        )
        save_all_collected_coins = max(n_coins, 1)
        self.summary_writer.add_scalar(
            f"{log_name}/steps_to_collect/all",
            all_steps / save_all_collected_coins,
            current_episode,
        )


if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
    # from pettingzoo.test import parallel_api_test

    env = parallel_env(
        render_mode="human",
        nb_players=3,
        grid_size=4,
        max_cycles=1000,
        randomize_coin=True,
    )
    # parallel_api_test(env, num_cycles=1000)

    # Reset the environment and get the initial observation
    obs = env.reset()
    nb_agent = 3
    # Run the environment for 10 steps
    for _ in range(1000):
        # Sample a random action
        actions = {"player_" + str(i): np.random.randint(4) for i in range(nb_agent)}

        #     # Step the environment and get the reward, observation, and done flag
        observations, rewards, terminations, truncations, infos = env.step(actions)

        #     # Print the reward
        # print(rewards)

        #     # If the game is over, reset the environment
        if terminations["player_0"]:
            obs = env.reset()
