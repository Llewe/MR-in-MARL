import functools
import logging
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame
import torch
from gymnasium.spaces import Box, Discrete, Space
from pettingzoo.utils import agent_selector as AgentSelector, wrappers
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
from torch.utils.tensorboard import SummaryWriter


def env(**kwargs):
    env = CoinGame(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def parallel_env(**kwargs):
    env = CoinGame(**kwargs)
    return env


SEED = 42


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NONE = 4


@dataclass(slots=True)
class AgentState:
    x: int
    y: int


@dataclass(slots=True)
class CoinState:
    x: int
    y: int
    owner: AgentID


@dataclass(slots=True)
class GlobalState:
    agent_states: dict[AgentID, AgentState]
    agent_to_id: dict[AgentID, int]
    coin_state: CoinState

    coin_is_collected: bool
    steps_on_board: int

    _obs: dict[AgentID, np.ndarray]

    _global_obs: np.ndarray | None

    def cal_obs(self) -> None:
        for agent_id in self.agent_states:
            self._obs[agent_id] = self.to_obs_list(agent_id)
        self._global_obs = self.to_obs_list(None)

    def to_obs_list(self, agent_id: AgentID | None) -> np.ndarray:
        """

        Parameters
        ----------
        agent_id: AgentID
            the agent for which the observation should be generated
        Returns
        -------
        np.ndarray[int]
            All information about the environment that are visible to the agent
            [0] agent_0.x, [1] agent_0.y,
            [2] agent_1.x, [3] agent_1.y,
            [4] coin.x, [5] coin.y,
            [6] coin.owner (1 if agent is owner, otherwise 0)

        """

        agent_state_values = [
            np.array([coord])
            for agent in self.agent_states.values()
            for coord in (agent.x, agent.y)
        ]
        return np.asarray(
            agent_state_values
            + [
                np.array([self.coin_state.x]),
                np.array([self.coin_state.y]),
                np.array(
                    [
                        int(
                            self.coin_state.owner == agent_id
                            if agent_id is not None
                            else self.agent_to_id[self.coin_state.owner]
                        )
                    ]
                ),
            ]
        )

    def get_obs(self, agent_id: AgentID) -> np.ndarray:
        return self._obs[agent_id]

    def get_global_obs(self) -> np.ndarray:
        return self._global_obs.copy()

    def get_all_obs(self) -> dict[AgentID, np.ndarray]:
        return self._obs


@dataclass(slots=True)
class HistoryState:
    board_step: int
    rewards: dict[AgentID, float]
    collected_coins: dict[AgentID, bool]
    coin_owner: AgentID
    actions: dict[AgentID, Action]
    pos: dict[AgentID, tuple[int, int]]
    pos_coin: tuple[int, int]


class CoinGamePygameRenderer:
    dark_mode: bool = False

    width: int
    height: int
    grid_size: int
    n_players: int
    screen: pygame.Surface

    colour_background: Tuple[int, int, int]
    color_grid: Tuple[int, int, int]
    color_players: List[Tuple[int, int, int]]
    color_coin: Tuple[int, int, int]

    def __init__(self, width: int, height: int, grid_size: int, n_players: int):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.n_players = n_players
        pygame.init()
        self.surface = pygame.Surface([self.width, self.height])
        self.screen = pygame.display.set_mode(self.surface.get_size())

        if self.dark_mode:
            self.colour_background = (48, 48, 48)
            self.color_grid = (255, 255, 255)
            self.color_players = [
                (233, 30, 99),
                (156, 39, 176),
                (103, 58, 183),
                (63, 81, 181),
                (33, 150, 243),
                (0, 188, 212),
                (0, 150, 136),
            ]
            self.color_coin = (255, 0, 0)
        else:
            self.colour_background = (255, 255, 255)
            self.color_grid = (200, 200, 200)
            self.color_players = [
                (233, 30, 99),
                (156, 39, 176),
                (103, 58, 183),
                (63, 81, 181),
                (33, 150, 243),
                (0, 188, 212),
                (0, 150, 136),
            ]
            self.color_coin = (255, 0, 0)

    def draw(self, global_state: GlobalState) -> None:
        # clear screen
        self.screen.fill(self.colour_background)

        # draw grid
        cell_size_w = self.width / self.grid_size
        cell_size_h = self.height / self.grid_size

        player_size = (
            0.8 * cell_size_w / 2
            if cell_size_w < cell_size_h
            else 0.8 * cell_size_h / 2
        )
        size_coin = (
            0.4 * cell_size_w / 2
            if cell_size_w < cell_size_h
            else 0.4 * cell_size_h / 2
        )

        for w in range(1, self.grid_size):
            pos_w = w * cell_size_w

            for h in range(1, self.grid_size):
                pos_h = h * cell_size_h
                pygame.draw.line(
                    self.screen,
                    self.color_grid,
                    (pos_w, 0),
                    (pos_w, self.height),
                )
                pygame.draw.line(
                    self.screen,
                    self.color_grid,
                    (0, pos_h),
                    (self.width, pos_h),
                )
        player_degree = 2 * math.pi / self.n_players
        # draw players
        for i, (agent_id, pos) in enumerate(global_state.agent_states.items()):
            assert isinstance(pos, AgentState)

            p_x = pos.x * cell_size_w + cell_size_w / 2
            p_y = pos.y * cell_size_h + cell_size_h / 2

            rec = pygame.Rect(
                p_x - player_size, p_y - player_size, player_size * 2, player_size * 2
            )

            player_degree_start = i * player_degree
            player_degree_end = (i + 1) * player_degree
            pygame.draw.arc(
                self.screen,
                self.color_players[i],
                rec,
                start_angle=player_degree_start,
                stop_angle=player_degree_end,
                width=20,
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet

            if global_state.coin_state.owner is agent_id:
                # draw coins
                c_x = global_state.coin_state.x * cell_size_w + cell_size_w // 2
                c_y = global_state.coin_state.y * cell_size_h + cell_size_h // 2

                pygame.draw.circle(
                    self.screen,
                    self.color_players[i],
                    (c_x, c_y),
                    radius=size_coin,
                )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet

        pygame.display.flip()


class CoinGame(ParallelEnv):
    metadata: Dict = {
        "render_modes": ["human"],
        "name": "coin_game_llewe",
        "is_parallelizable": True,
    }
    # Coingame Variables
    with_none_action: bool
    walls: bool
    max_cycles: int
    n_players: int
    randomize_coin: bool
    allow_overlap_players: bool
    name: str = "coin_game_llewe"
    nr_actions: int

    random: random.Random

    # Render Variables
    render_mode: str
    pygame_renderer: Optional[CoinGamePygameRenderer]

    # Log Variables
    summary_writer: Optional[SummaryWriter]

    # Internal Variables
    global_state: GlobalState

    # Action, State Log
    current_history: List[HistoryState]

    # Execution Variables
    agent_selector: AgentSelector

    def __init__(
        self,
        with_none_action: bool = True,
        walls: bool = False,
        max_cycles: int = 150,
        render_mode: str = "",
        n_players: int = 2,
        grid_size: int = 3,
        randomize_coin: bool = True,
        allow_overlap_players: bool = False,
        summary_writer: Optional[SummaryWriter] = None,
    ):
        """

        Parameters
        ----------
        with_none_action: bool
            if "None" action is allowed
        walls: bool
            whether to include walls in the environment. If false players moving
            outside the grid will be placed on the opposite side
        max_cycles: int
        render_mode: str
        n_players: int
        grid_size: int
        randomize_coin: bool
        allow_overlap_players: bool
        summary_writer: Optional[SummaryWriter]
        """

        super().__init__()
        self.random = random.Random(42)
        self.with_none_action = with_none_action
        self.walls = walls
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.n_players = n_players
        self.grid_size = grid_size
        self.randomize_coin = randomize_coin
        self.allow_overlap_players = allow_overlap_players
        self.summary_writer = summary_writer

        self.current_history: list[HistoryState] = []

        if render_mode == "human":
            self.pygame_renderer = CoinGamePygameRenderer(
                width=700, height=700, grid_size=grid_size, n_players=n_players
            )

        # Configure Agents
        self.agents: list[AgentID] = [f"player_{r}" for r in range(self.n_players)]
        self.possible_agents: list[AgentID] = self.agents[:]

        self.global_state = GlobalState(
            agent_states={
                agent: AgentState(x=1, y=2) for agent in self.possible_agents
            },
            agent_to_id={agent: i for i, agent in enumerate(self.possible_agents)},
            coin_state=CoinState(x=3, y=4, owner=self.possible_agents[0]),
            coin_is_collected=False,
            steps_on_board=0,
            _global_obs=None,
            _obs={},
        )

        self.nr_actions = len(Action) if with_none_action else len(Action) - 1
        self.action_spaces = {
            agent: Discrete(self.nr_actions) for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: Box(
                low=0,
                high=n_players if n_players > grid_size else grid_size,
                shape=(len(self.global_state.to_obs_list(agent)), 1),
                dtype=np.int64,
            )
            for agent in self.possible_agents
        }

        self.rewards: dict[AgentID, float] = {
            agent: 0 for agent in self.possible_agents
        }

    def get_global_obs(self) -> ObsType:
        return self.global_state.get_global_obs()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> Space:
        return self.action_spaces[agent]

    def _all_locations(self) -> list[tuple[int, int]]:
        return [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]

    def _get_unoccupied_locations(self) -> list[tuple[int, int]]:
        pos: list[tuple[int, int]] = self._all_locations()

        for agent in self.global_state.agent_states.values():
            if (agent.x, agent.y) in pos:
                pos.remove((agent.x, agent.y))
        return pos

    def _generate_coin(
        self, possible_pos: Optional[list[tuple[int, int]]] = None
    ) -> None:
        """
        Generate the coin at a random location that is not occupied by an agent
        Parameters
        ----------
        possible_pos: Optional[list[tuple[int, int]]]
            if set to None, all unoccupied locations will be considered
            if set one of the possible positions will be chosen
            (no check if it is occupied)

        Returns
        -------

        """
        if possible_pos is None:
            possible_pos = self._get_unoccupied_locations()

        for agent in self.global_state.agent_states.values():
            if (agent.x, agent.y) in possible_pos:
                possible_pos.remove((agent.x, agent.y))

        (
            self.global_state.coin_state.x,
            self.global_state.coin_state.y,
        ) = self.random.choice(possible_pos)

        if self.randomize_coin:
            self.global_state.coin_state.owner = self.random.choice(
                self.possible_agents
            )

        self.global_state.coin_is_collected = False

    def _reset_board(self) -> None:
        """
        Reset the board. All agents will be placed at a random location. If the Agents
        can overlap they can also be placed on the same location.
        The coin will allways be placed at a random location that is not occupied
        by an agent.
        Returns
        -------

        """
        possible_pos: list[tuple[int, int]] = self._all_locations()

        for agent_state in self.global_state.agent_states.values():
            agent_state.x, agent_state.y = self.random.choice(possible_pos)

            if not self.allow_overlap_players:
                possible_pos.remove((agent_state.x, agent_state.y))

        self._generate_coin(possible_pos=possible_pos)

        self.global_state.steps_on_board = 0

    def reinit(self, options=None) -> None:
        self.agents = self.possible_agents[:]
        self.agent_selector: AgentSelector = AgentSelector(self.possible_agents)

        self.agent_selection: AgentID = self.agent_selector.next()

        self._clear_rewards()

        self._cumulative_rewards = {name: 0.0 for name in self.possible_agents}
        self.terminations: dict[AgentID, bool] = {
            agent: False for agent in self.possible_agents
        }
        self.truncations: dict[AgentID, bool] = {
            agent: False for agent in self.possible_agents
        }
        self.infos: dict[AgentID, dict[str, Any]] = {
            agent: {} for agent in self.possible_agents
        }

        self._reset_board()
        self.global_state.cal_obs()

    def _clear_rewards(self) -> None:
        self.rewards = {name: 0.0 for name in self.possible_agents}

    def _render_text(self) -> None:
        raise NotImplementedError("Text rendering is not implemented yet.")

    def _render_pygame(self) -> None:
        if self.pygame_renderer is not None:
            self.pygame_renderer.draw(self.global_state)

    def render(self) -> None:
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            logging.warning(
                "You are calling render method without specifying any render mode."
            )
            return
        if self.render_mode == "text":
            self._render_text()
        elif self.render_mode == "human":
            self._render_pygame()

    def observe(self, agent: AgentID) -> ObsType | None:
        return self.global_state.get_obs(agent)

    def close(self) -> None:
        pass

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        if seed is not None:
            self.random.seed(seed)
        self.reinit(options=options)
        if options is None:
            self.current_history.clear()
            logging.warning("Resetting history. No options in reset -> no logging.")
        elif "history_reset" in options and options["history_reset"]:
            self.current_history.clear()
        else:
            self.log(options)
        return self.global_state.get_all_obs().copy(), self.infos

    def _check_bounds(self, new_pos: int) -> int:
        if self.walls:
            if new_pos < 0:
                new_pos = 0
            elif new_pos >= self.grid_size:
                new_pos = self.grid_size - 1
        else:
            new_pos = new_pos % self.grid_size
        return new_pos

    def _move_reward_agent(self, agent: AgentID, action: Action) -> None:
        """
        Move the agent according to the action. If a coin is collected, the coin will be
        placed at a random location that is not occupied by an agent.
        Parameters
        ----------
        agent: AgentID
        action: Action

        Returns
        -------
        int
            reward
        """
        pos_x: int = self.global_state.agent_states[agent].x
        pos_y: int = self.global_state.agent_states[agent].y
        # Calculate next position
        if action == Action.LEFT:
            pos_x = self._check_bounds(pos_x - 1)
        elif action == Action.RIGHT:
            pos_x = self._check_bounds(pos_x + 1)
        elif action == Action.UP:
            pos_y = self._check_bounds(pos_y + 1)
        elif action == Action.DOWN:
            pos_y = self._check_bounds(pos_y - 1)
        elif action == Action.NONE:
            pass
        else:
            # Handle any other cases, if needed
            pass

        # Check if the new position is occupied by another agent
        if not self.allow_overlap_players:
            for agent_state in self.global_state.agent_states.values():
                if (agent_state.x, agent_state.y) == (pos_x, pos_y):
                    # position is occupied, stay still
                    return

        self.global_state.agent_states[agent].x = pos_x
        self.global_state.agent_states[agent].y = pos_y

        # Check if the new position is occupied by a coin :)
        if (pos_x, pos_y) == (
            self.global_state.coin_state.x,
            self.global_state.coin_state.y,
        ):
            self.current_history[-1].collected_coins[agent] = True
            self.global_state.coin_is_collected = True
            self.rewards[agent] += 1.0
            if self.global_state.coin_state.owner is not agent:
                self.rewards[self.global_state.coin_state.owner] -= 2.0

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],  # observations
        dict[AgentID, float],  # rewards
        dict[AgentID, bool],  # dones
        dict[AgentID, bool],  # truncations
        dict[AgentID, dict],  # infos
    ]:
        """Receives a dictionary of actions keyed by the agent name.

        Returns the observation dictionary, reward dictionary, terminated dictionary, truncated dictionary
        and info dictionary, where each dictionary is keyed by the agent.
        """

        self._clear_rewards()
        self.current_history.append(
            HistoryState(
                board_step=self.global_state.steps_on_board,
                rewards={},
                collected_coins={a: False for a in self.possible_agents},
                coin_owner=self.global_state.coin_state.owner,
                actions={},
                pos={a: (s.x, s.y) for a, s in self.global_state.agent_states.items()},
                pos_coin=(
                    self.global_state.coin_state.x,
                    self.global_state.coin_state.y,
                ),
            )
        )
        # Step all agents

        agents_order = self.agents[:]
        random.shuffle(agents_order)
        for agent in agents_order:
            action = actions[agent]

            if self.truncations[agent]:
                continue
            self._move_reward_agent(agent, Action(action))

            # Agent steps are done
        if self.global_state.coin_is_collected:
            self._generate_coin()

        self.global_state.steps_on_board += 1

        env_truncation = self.global_state.steps_on_board >= self.max_cycles

        self.truncations = {agent: env_truncation for agent in self.possible_agents}
        if env_truncation:
            self.agents = []

        # update observations
        self.global_state.cal_obs()

        for a, r in self.rewards.items():
            self.current_history[-1].rewards[a] = r

        if self.render_mode != "":
            self.render()

        return (
            self.global_state.get_all_obs().copy(),
            self.rewards,
            self.truncations,  # we don't use terminations
            self.truncations,
            self.infos,
        )

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

        if "heatmap" in options:
            heatmap = options["heatmap"]

        log_name = f"coin_game-{tag}"

        cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        cnt_collected_coins = {agent: 0.0 for agent in self.possible_agents}

        total_collected_coins: float = 0.0
        total_coins_collected_by_owner: float = 0.0

        collected_coins_owner = {
            agent: {agent: 0.0 for agent in self.agents}
            for agent in self.possible_agents
        }

        divider: float = (
            1.0  # This is used if multiple episodes are logged at once (e.g. one epoch)
        )

        if len(self.current_history) > self.max_cycles:
            divider = len(self.current_history) // self.max_cycles
            if divider * self.max_cycles != len(self.current_history):
                raise ValueError(
                    "The length of the current history is not a multiple of the max_cycles."
                )

        for history in self.current_history:
            for agent_id in self.possible_agents:
                cumulative_rewards[agent_id] += history.rewards[agent_id]
                cnt_collected_coins[agent_id] += history.collected_coins[agent_id]
                if history.collected_coins[agent_id]:
                    collected_coins_owner[agent_id][history.coin_owner] += 1.0

                    total_collected_coins += 1.0
                    if agent_id == history.coin_owner:
                        total_coins_collected_by_owner += 1.0

        if divider != 1.0:
            cumulative_rewards = {a: r / divider for a, r in cumulative_rewards.items()}
            total_collected_coins /= divider
            total_coins_collected_by_owner /= divider

            cnt_collected_coins = {
                a: c / divider for a, c in cnt_collected_coins.items()
            }
            collected_coins_owner = {
                aa: {a: c / divider for a, c in cco.items()}
                for aa, cco in collected_coins_owner.items()
            }

        for agent_id in self.possible_agents:
            self.summary_writer.add_scalar(
                f"{log_name}/cumulative_reward/{agent_id}",
                cumulative_rewards[agent_id],
                epoch,
            )
            self.summary_writer.add_scalar(
                f"{log_name}/collected_coins/{agent_id}",
                cnt_collected_coins[agent_id],
                epoch,
            )

            for owner, cnt in collected_coins_owner[agent_id].items():
                self.summary_writer.add_scalar(
                    f"{log_name}/collected_coins_owner/{agent_id}/{owner}",
                    cnt,
                    epoch,
                )

        self.summary_writer.add_scalar(
            f"{log_name}/coins/total/",
            total_collected_coins,
            epoch,
        )

        own_coin: float
        if total_collected_coins > 0:
            own_coin = total_coins_collected_by_owner / total_collected_coins
        else:
            own_coin = 0.0

        self.summary_writer.add_scalar(f"{log_name}/coins/own_coin/", own_coin, epoch)

        # calculate position histogram

        if heatmap:
            coin_pos = np.zeros((self.grid_size, self.grid_size))
            agent_pos = {
                agent: np.zeros((self.grid_size, self.grid_size))
                for agent in self.possible_agents
            }
            for history in self.current_history:
                coin_pos[history.pos_coin[0], history.pos_coin[1]] += 1
                for agent_id, pos in history.pos.items():
                    agent_pos[agent_id][pos[0], pos[1]] += 1

            coin_pos /= len(self.current_history)
            image_data = torch.from_numpy(coin_pos).unsqueeze(0).unsqueeze(0)

            # Log the 2D position as an image
            self.summary_writer.add_image(
                f"{log_name}/heatmap/coin",
                image_data,
                global_step=epoch,
                dataformats="NCHW",
            )

            for agent_id in self.possible_agents:
                agent_pos[agent_id] /= len(self.current_history)
                # Convert the NumPy array to a torch.Tensor and add batch and channel dimensions
                agent_image_data = (
                    torch.from_numpy(agent_pos[agent_id]).unsqueeze(0).unsqueeze(0)
                )
                # Log the 2D position as an image
                self.summary_writer.add_image(
                    f"{log_name}/heatmap/{agent_id}",
                    agent_image_data,
                    global_step=epoch,
                    dataformats="NCHW",
                )

        self.current_history.clear()


if __name__ == "__main__":
    from pettingzoo.test.parallel_test import parallel_api_test

    parallel_api_test(parallel_env(), num_cycles=1000)
