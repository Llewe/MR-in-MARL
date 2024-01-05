# reference
# https://github.com/arjun-prakash/pz_dilemma
# https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/rps/rps.py
# Modified to work better with tensorboard and some minor code cleanup changes.
import logging
from dataclasses import dataclass
from typing import List

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.utils.env import ActionType, AgentID
from torch.utils.tensorboard import SummaryWriter

from src.envs.dilemma.games import (
    Chicken,
    Game,
    Prisoners_Dilemma,
    Samaritans_Dilemma,
    Stag_Hunt,
)


def env(**kwargs):
    env = DilemmaEnv(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


@dataclass(slots=True)
class HistoryState:
    board_step: int
    rewards: dict[AgentID, float]
    actions: dict[AgentID, ActionType]
    winner: AgentID | None = None


class DilemmaEnv(AECEnv):
    """Two-player environment for rock paper scissors.
    Expandable environment to rock paper scissors lizard spock action_6 action_7 ...
    The observation is simply the last opponent action.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "simple_pd_v0",
        "is_parallelizable": True,
    }

    # Action, State Log
    current_history: List[HistoryState]
    summary_writer: SummaryWriter

    name: str

    def __init__(
        self,
        game: str = "pd",
        num_actions: int = 2,
        max_cycles: int = 15,
        render_mode: str = "",
        summary_writer: SummaryWriter | None = None,
    ):
        self.max_cycles = max_cycles
        GAMES: dict[str, Game] = {
            "pd": Prisoners_Dilemma(),
            "sd": Samaritans_Dilemma(),
            "stag": Stag_Hunt(),
            "chicken": Chicken(),
        }
        self.summary_writer = summary_writer
        self.current_history = []
        self.render_mode = render_mode
        self.name = game
        self.game = GAMES[game]

        self._moves = self.game.moves
        # none is last possible action, to satisfy discrete action space
        self._none = self.game.NONE

        self.agents = ["player_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.action_spaces = {agent: Discrete(num_actions) for agent in self.agents}
        self.observation_spaces = {
            agent: Discrete(num_actions) for agent in self.agents
        }

        self.reinit()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reinit(self):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.state = {agent: self._none for agent in self.agents}
        self.observations = {
            agent: [self._none] * len(self.possible_agents) for agent in self.agents
        }

        self.history = [0] * (2 * 5)

        self.num_moves = 0

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

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                self._moves[self.state[self.agents[0]]],
                self._moves[self.state[self.agents[1]]],
            )
        else:
            string = "Game over"
        print(string)

    def observe(self, agent):
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        pass

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        self.reinit()

        if options is None:
            self.current_history.clear()
            logging.warning("Resetting history. No options in reset -> no logging.")
        elif "history_reset" in options and options["history_reset"]:
            self.current_history.clear()
        else:
            self.log(options)

    def step(self, action: ActionType) -> None:
        agent: AgentID = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        if self._agent_selector.is_first():
            self.current_history.append(
                HistoryState(
                    board_step=self.num_moves,
                    rewards={},
                    actions={},
                    winner=None,
                )
            )
        self.current_history[-1].actions[agent] = action  # Log action

        self.state[agent] = action

        # collect reward if it is the last agent to act

        if self._agent_selector.is_last():
            (
                self.rewards[self.agents[0]],
                self.rewards[self.agents[1]],
            ) = self.game.payoff[
                (self.state[self.agents[0]], self.state[self.agents[1]])
            ]

            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_cycles for agent in self.agents
            }

            # observe the current state
            for i in self.agents:
                self.observations[i] = list(
                    self.state.values()
                )  # TODO: consider switching the board
            agent_0: AgentID = self.agents[0]
            agent_1: AgentID = self.agents[1]

            reward_0 = self.rewards[agent_0]
            reward_1 = self.rewards[agent_1]
            self.current_history[-1].rewards[agent_0] = reward_0
            self.current_history[-1].rewards[agent_1] = reward_1

            if reward_0 > reward_1:
                self.current_history[-1].winner = agent_0
            elif reward_1 > reward_0:
                self.current_history[-1].winner = agent_1
            else:
                self.current_history[-1].winner = None

        else:
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self._none
            self._clear_rewards()

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        if self.render_mode != "":
            self.render()

    def log(self, options: dict) -> None:
        """
        Log various episode metrics to tensorboard. This method should be called after every episode.
        Parameters
        ----------
        options: dict

        Returns
        -------

        """

        if self.summary_writer is None:
            self.current_history.clear()
            logging.warning("No summary writer. Not logging.")
            return

        if "write_log" not in options:
            self.current_history.clear()
            logging.warning("No write_log in options. Not logging.")
            return
        else:
            write_log = options["write_log"]

        if not write_log:
            return  # Log everything at another time

        epoch: int = 0
        tag: str = ""
        heatmap: bool = False

        if "epoch" in options:
            epoch = options["epoch"]

        if "tag" in options:
            tag = options["tag"]

        log_name = f"{self.name}-{tag}"

        divider: float = (
            1.0  # This is used if multiple episodes are logged at once (e.g. one epoch)
        )
        history_len = len(self.current_history)
        if history_len > self.max_cycles:
            divider = history_len // self.max_cycles
            if divider * self.max_cycles != history_len:
                raise ValueError(
                    "The length of the current history is not a multiple of the max_cycles."
                )

        reward_p0: float = 0.0
        reward_p1: float = 0.0

        winner_p0: float = 0.0
        winner_p1: float = 0.0

        for history in self.current_history:
            reward_p0 += history.rewards["player_0"]
            reward_p1 += history.rewards["player_1"]

            if history.winner == "player_0":
                winner_p0 += 1
            elif history.winner == "player_1":
                winner_p1 += 1

        if divider != 1.0:
            # scale rewards so they are comparable if the number of epochs changes
            # reward_p0 /= divider
            # reward_p1 /= divider
            pass

        efficiency = reward_p0 + reward_p1

        efficiency_per_step = efficiency / history_len

        self.summary_writer.add_scalar(
            f"{log_name}/efficiency_per_step", efficiency_per_step, epoch
        )

        # p0 win == 1, p1 win == -1, draw == 0
        wins_p0_per_step = (winner_p0 - winner_p1) / history_len
        self.summary_writer.add_scalar(
            f"{log_name}/wins_p0_per_step", wins_p0_per_step, epoch
        )

        self.current_history.clear()


if __name__ == "__main__":
    SEED = 0
    if SEED is not None:
        np.random.seed(SEED)
    # from pettingzoo.test import parallel_api_test

    env = parallel_env(render_mode="human")
    # parallel_api_test(env, num_cycles=1000)

    # Reset the environment and get the initial observation
    obs = env.reset()
    print(env.observation_spaces)
    # Run the environment for 10 steps
    for _ in range(10):
        # Sample a random action
        actions = {"player_" + str(i): np.random.randint(2) for i in range(2)}

        # Step the environment and get the reward, observation, and done flag
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(observations)
        # Print the reward
        # print(rewards)
        print("observations: ", observations)
        # If the game is over, reset the environment
        if terminations["player_0"]:
            obs = env.reset()
