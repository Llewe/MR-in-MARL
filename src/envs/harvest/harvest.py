import functools
import logging
import random
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pygame
import torch
from gymnasium.spaces import Box, Discrete, Space
from pettingzoo.utils.env import ActionType, AgentID, ObsType, ParallelEnv
from torch.utils.tensorboard import SummaryWriter


def parallel_env(**kwargs):
    env = Harvest(**kwargs)
    return env


class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    NONE = 4

    TAG_N = 5
    TAG_E = 6
    TAG_S = 7
    TAG_W = 8


class ObsTiles(Enum):
    EMPTY = 0
    AGENT = 1
    APPLE = 2
    SELF = 3
    OUTSIDE = 4


@dataclass(slots=True)
class HistoryState:
    board_step: int
    rewards: dict[AgentID, float]
    collected_apples: dict[AgentID, int]
    tagged_agents: dict[AgentID, int]
    apples_on_board: int
    actions: dict[AgentID, Action]
    agents_pos: dict[AgentID, tuple[int, int]]


@dataclass(slots=True)
class AgentState:
    x: int
    y: int

    tagged: int = 0

    def isTagged(self):
        return self.tagged > 0

    def reset(self):
        self.tagged = 0


@dataclass(slots=True)
class Apple:
    x: int
    y: int


@dataclass(slots=True)
class GlobalState:
    agent_states: Dict[AgentID, AgentState]

    apples: np.ndarray

    # outside_vision_range = 2 * vision_range + 1 (normal vision is from the agent's position to x tiles in each direction)
    vision_range: int
    # outside_beam_range = 2 * tag_beam_width + 1 (normal beam is from the agent's position to x tiles in each direction)
    tag_beam_width: int

    map_width: int
    map_height: int

    vision_offsets: List[Tuple[int, int, int]]

    # Walls = -1, Empty = 0, Agent = 1, Apple = 2
    _obs: dict[AgentID, np.ndarray]
    _neighbours: dict[AgentID, List[AgentID]]

    fixed_spawn: bool

    steps_on_board: int = 0

    """
         0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4
         1 . . . . A . . . . A . . . . A . . . . A . . . .
         2 . . . A A A . . A A A . . A A A . . A A A . . .
         3 . . . . A . . . . A . . . . A . . . . A . . . .
         4 . . . . . . . . . . . . . . . . . . . . . . . .
         5 . A . . . . A . . . . A . . . . A . . . . A . .
         6 A A A . . A A A . . A A A . . A A A . . A A A .
         7 . A . . . . A . . . . A . . . . A . . . . A . .
         8 . . . . . . . . . . . . . . . . . . . . . . . .
        """
    fixed_spawn_positions = [
        (5, 1),
        (10, 1),
        (15, 1),
        (20, 1),
        (4, 2),
        (5, 2),
        (6, 2),
        (9, 2),
        (10, 2),
        (11, 2),
        (14, 2),
        (15, 2),
        (16, 2),
        (19, 2),
        (20, 2),
        (21, 2),
        (5, 3),
        (10, 3),
        (15, 3),
        (20, 3),
        (2, 5),
        (7, 5),
        (12, 5),
        (17, 5),
        (22, 5),
        (1, 6),
        (2, 6),
        (3, 6),
        (6, 6),
        (7, 6),
        (8, 6),
        (11, 6),
        (12, 6),
        (13, 6),
        (16, 6),
        (17, 6),
        (18, 6),
        (21, 6),
        (22, 6),
        (23, 6),
        (2, 7),
        (7, 7),
        (12, 7),
        (17, 7),
        (22, 7),
    ]

    def __init__(
        self,
        agents: List[AgentID],
        map_width: int = 25,
        map_height: int = 9,
        vision_range: int = 3,
        tag_beam_width: int = 2,
        fixed_spawn: bool = True,
    ):
        self.agent_states = {}
        self._obs = {}
        self._neighbours = {}
        self.map_width = map_width
        self.map_height = map_height
        self.vision_range = vision_range
        self.tag_beam_width = tag_beam_width
        self.fixed_spawn = fixed_spawn

        outside_vision_range = 2 * self.vision_range + 1
        for agent in agents:
            self.agent_states[agent] = AgentState(0, 0, 0)
            self._obs[agent] = np.zeros(
                (outside_vision_range * outside_vision_range, 1), dtype=np.int64
            )
            self._neighbours[agent] = []

        self.vision_offsets = [
            (index, x, y)
            for index, (x, y) in enumerate(
                (
                    (x, y)
                    for x, y in product(
                        range(-self.vision_range, self.vision_range + 1),
                        range(-self.vision_range, self.vision_range + 1),
                    )
                    if not (x == 0 and y == 0)
                ),
            )
        ]
        self.reset_apple_map(None)

    def pos_to_obs_index(self, x: int, y: int) -> int:
        return x * self.vision_range + y

    def in_map(self, x: int, y: int) -> bool:
        return 0 <= x < self.map_width and 0 <= y < self.map_height

    def in_vision(self, agent: AgentState, obj: Union[Apple, AgentState]):
        return (
            abs(obj.x - agent.x) <= self.vision_range
            and abs(obj.y - agent.y) <= self.vision_range
        )

    def get_neighbours(self) -> dict[AgentID, List[AgentID]]:
        return self._neighbours

    def cal_obs(self) -> None:
        curr_map: np.ndarray = self.build_map()
        for agent_id, agent in self.agent_states.items():
            self._neighbours[agent_id].clear()
            for index, x, y in self.vision_offsets:
                if not self.in_map(x, y):
                    self._obs[agent_id][index] = ObsTiles.OUTSIDE.value
                else:
                    if x == agent.x and y == agent.y:
                        self._obs[agent_id][index] = ObsTiles.SELF.value
                    else:
                        self._obs[agent_id][index] = curr_map[x, y]

                    if curr_map[x, y] == ObsTiles.AGENT.value:
                        for id, a in self.agent_states.items():
                            if a.x == x and a.y == y and id != agent_id:
                                self._neighbours[agent_id].append(id)

    def build_map(self) -> np.ndarray:
        curr_map = np.copy(self.apples)

        for a in self.agent_states.values():
            curr_map[a.x, a.y] = ObsTiles.AGENT.value
        return curr_map

    def reset_apple_map(self, apple_pos: Optional[List[Tuple[int, int]]]) -> None:
        self.apples = np.zeros(shape=(self.map_width, self.map_height), dtype=np.int32)
        if apple_pos is not None:
            for x, y in apple_pos:
                self.apples[x, y] = 1

    def apples_on_board(self) -> int:
        return np.sum(self.apples)

    def get_obs(self, agent_id: AgentID) -> np.ndarray:
        return self._obs[agent_id]

    @functools.lru_cache(maxsize=None)
    def get_beam_size(self) -> int:
        return 2 * self.tag_beam_width + 1

    def get_all_obs(self) -> dict[AgentID, np.ndarray]:
        return self._obs


class HarvestPygameRenderer:
    grid_width: int = 18
    grid_height: int = 10

    margin_player: int = 10
    margin_apple: int = 2

    cell_size: int

    n_players: int
    screen: pygame.Surface

    colour_background = (16, 16, 16)
    color_grid = (0, 0, 0)
    color_players = [
        (233, 30, 99),
        (156, 39, 176),
        (103, 58, 183),
        (63, 81, 181),
        (33, 150, 243),
        (0, 188, 212),
        (0, 150, 136),
        (233, 30, 99),
        (156, 39, 176),
        (103, 58, 183),
        (63, 81, 181),
        (33, 150, 243),
        (0, 188, 212),
        (0, 150, 136),
        (233, 30, 99),
        (156, 39, 176),
        (103, 58, 183),
        (63, 81, 181),
        (33, 150, 243),
        (0, 188, 212),
        (0, 150, 136),
    ]
    apple_color = (0, 128, 0)

    def __init__(self, grid_width: int, grid_height: int, n_players: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.n_players = n_players

        self.cell_size = 64

        pygame.init()
        self.screen = pygame.display.set_mode(
            pygame.Surface(
                [self.grid_width * self.cell_size, self.grid_height * self.cell_size]
            ).get_size()
        )

    def _draw_grid(self) -> None:
        for x, y in np.ndindex((self.grid_width, self.grid_height)):
            pygame.draw.rect(
                self.screen,
                self.color_grid,
                (
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                ),
                1,
            )

    def _draw_apples(self, state: GlobalState) -> None:
        for x, y in product(range(state.map_width), range(state.map_height)):
            if state.apples[x, y] == 1:
                pygame.draw.rect(
                    self.screen,
                    self.apple_color,
                    (
                        x * self.cell_size + self.margin_apple,
                        y * self.cell_size + self.margin_apple,
                        self.cell_size - 2 * self.margin_apple,
                        self.cell_size - 2 * self.margin_apple,
                    ),
                )

    def _draw_agents(self, state: GlobalState) -> None:
        for agent_id, agent_state in state.agent_states.items():
            pygame.draw.circle(
                self.screen,
                self.color_players[int(agent_id[-1])],
                (
                    agent_state.x * self.cell_size + self.cell_size // 2,
                    agent_state.y * self.cell_size + self.cell_size // 2,
                ),
                self.cell_size // 2 - self.margin_player,
            )

    def draw(self, state: GlobalState) -> None:
        # clear screen
        self.screen.fill(self.colour_background)
        self._draw_grid()
        self._draw_apples(state)
        self._draw_agents(state)

        pygame.display.flip()


class Harvest(ParallelEnv):
    metadata: Dict = {
        "render_modes": ["human"],
        "name": "harvest_llewe",
        "is_parallelizable": True,
    }

    # Internal Variables
    global_state: GlobalState
    max_cycles: int
    init_apples: int
    regrow_chance: float

    tag_time: int

    # Action, State Log
    current_history: List[HistoryState]

    # Log Variables
    summary_writer: Optional[SummaryWriter]

    def __init__(
        self,
        render_mode="",
        max_cycles: int = 250,
        n_players: int = 6,
        n_apples: int = 12,
        grid_width: int = 18,
        grid_height: int = 10,
        vision_range: int = 3,
        tag_beam_width: int = 2,
        tag_time: int = 25,
        regrow_chance: float = 0.001,
        summary_writer: Optional[SummaryWriter] = None,
        fixed_spawn: bool = True,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_cycles = max_cycles
        self.n_players = n_players
        self.init_apples = n_apples
        self.tag_time = tag_time
        self.regrow_chance = regrow_chance
        self.agents: list[AgentID] = [f"player_{r}" for r in range(self.n_players)]
        self.possible_agents: list[AgentID] = self.agents[:]
        self.nr_actions = len(Action)
        self.action_spaces = {
            agent: Discrete(self.nr_actions) for agent in self.possible_agents
        }
        self.summary_writer = summary_writer

        self.current_history: list[HistoryState] = []

        self.observation_spaces = {
            agent: Box(
                low=min(item.value for item in ObsTiles),
                high=max(item.value for item in ObsTiles) + 1,
                shape=((vision_range * 2 + 1) * (vision_range * 2 + 1), 1),
                dtype=np.int64,
            )
            for agent in self.possible_agents
        }

        self.global_state = GlobalState(
            agents=self.agents,
            map_width=grid_width,
            map_height=grid_height,
            vision_range=vision_range,
            tag_beam_width=tag_beam_width,
            fixed_spawn=fixed_spawn,
        )

        if render_mode == "human":
            self.pygame_renderer = HarvestPygameRenderer(
                grid_width, grid_height, n_players
            )

        self.rewards: dict[AgentID, float] = {
            agent: 0 for agent in self.possible_agents
        }

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        self.render_mode = render_mode

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID) -> Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID) -> Space:
        return self.action_spaces[agent]

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

    def _reset_board(self) -> None:
        # reset helpers/lists
        self.global_state.steps_on_board = 0

        # spawn player
        all_positions = list(
            product(
                range(self.global_state.map_width),
                range(self.global_state.map_height),
            )
        )

        # spawn apples
        if self.global_state.fixed_spawn:
            self.global_state.reset_apple_map(self.global_state.fixed_spawn_positions)

        else:
            apple_pos: List[Tuple[int, int]] = []
            for i in range(self.init_apples):
                pos = all_positions.pop(np.random.randint(0, len(all_positions)))
                apple_pos.append(pos)
            self.global_state.reset_apple_map(apple_pos)

        for agent in self.possible_agents:
            agent_state: AgentState = self.global_state.agent_states[agent]

            agent_state.reset()

            pos = all_positions.pop(np.random.randint(0, len(all_positions)))

            agent_state.x = pos[0]
            agent_state.y = pos[1]

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        if options is None:
            self.current_history.clear()
            logging.warning("Resetting history. No options in reset -> no logging.")
        elif "history_reset" in options and options["history_reset"]:
            self.current_history.clear()
        else:
            self.log(options)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}

        self._reset_board()

        self.global_state.cal_obs()

        self.infos = {
            agent: {
                "neighbours": neighbours,
            }
            for agent, neighbours in self.global_state.get_neighbours().items()
        }
        return self.global_state.get_all_obs().copy(), self.infos

    def _check_bounds_x(self, new_x: int) -> int:
        if new_x < 0:
            new_x = 0
        elif new_x >= self.global_state.map_width:
            new_x = self.global_state.map_width - 1
        return new_x

    def _check_bounds_y(self, new_y: int) -> int:
        if new_y < 0:
            new_y = 0
        elif new_y >= self.global_state.map_height:
            new_y = self.global_state.map_height - 1
        return new_y

    def _tag_agents(
        self, start_x: int, start_y: int, end_x: int, end_y: int, tagger_id: AgentID
    ) -> None:
        for agent_id, state in self.global_state.agent_states.items():
            if agent_id != tagger_id:
                if start_x <= state.x <= end_x and start_y <= state.y <= end_y:
                    state.tagged = self.tag_time

    def _move_reward_agent(self, agent: AgentID, action: Action) -> None:
        """
        Move the agent according to the action.
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
            pos_x = pos_x - 1
        elif action == Action.RIGHT:
            pos_x = pos_x + 1
        elif action == Action.UP:
            pos_y = pos_y + 1
        elif action == Action.DOWN:
            pos_y = pos_y - 1
        elif action == Action.NONE:
            pass

        elif action == Action.TAG_N:
            self._tag_agents(
                pos_x - self.global_state.tag_beam_width,
                pos_y,
                pos_x + self.global_state.tag_beam_width,
                pos_y + self.global_state.tag_beam_width,
                agent,
            )
            return
        elif action == Action.TAG_E:
            self._tag_agents(
                pos_x,
                pos_y - self.global_state.tag_beam_width,
                pos_x + self.global_state.tag_beam_width,
                pos_y + self.global_state.tag_beam_width,
                agent,
            )
            return
        elif action == Action.TAG_S:
            self._tag_agents(
                pos_x - self.global_state.tag_beam_width,
                pos_y - self.global_state.tag_beam_width,
                pos_x + self.global_state.tag_beam_width,
                pos_y,
                agent,
            )
            return
        elif action == Action.TAG_W:
            self._tag_agents(
                pos_x - self.global_state.tag_beam_width,
                pos_y - self.global_state.tag_beam_width,
                pos_x,
                pos_y + self.global_state.tag_beam_width,
                agent,
            )
            return

        if not self.global_state.in_map(pos_x, pos_y):
            return

        # Check if the new position is occupied by another agent
        for agent_state in self.global_state.agent_states.values():
            if (agent_state.x, agent_state.y) == (pos_x, pos_y):
                # position is occupied, stay still
                return

        self.global_state.agent_states[agent].x = pos_x
        self.global_state.agent_states[agent].y = pos_y

        if self.global_state.apples[pos_x, pos_y] == 1:
            self.global_state.apples[pos_x, pos_y] = 0
            self.current_history[-1].collected_apples[agent] += 1
            self.rewards[agent] += 1

    def _grow_apples_new(self) -> None:
        """
        Implements the apple regrowth mechanic. From MATE:
        Returns
        -------

        """
        agent_pos = [(a.x, a.y) for a in self.global_state.agent_states.values()]

        for x, y in product(
            range(self.global_state.map_width), range(self.global_state.map_height)
        ):
            if self.global_state.apples[x][y] == 0 and (x, y) not in agent_pos:
                apple_count = 0
                for offset_x, offset_y in product(range(-2, 3), range(-2, 3)):
                    pos_x, pos_y = x + offset_x, y + offset_y
                    if self.global_state.in_map(pos_x, pos_y):
                        distance = abs(offset_x) + abs(
                            offset_y
                        )  # remove "diagonal" apples
                        if (
                            distance <= 2
                            and self.global_state.apples[pos_x][pos_y] == 1
                        ):
                            apple_count += 1
                if apple_count > 0:
                    prob: float = 0
                    if apple_count in [1, 2]:
                        prob = 0.01
                    if apple_count in [3, 4]:
                        prob = 0.05
                    if apple_count > 4:
                        prob = 0.1
                    self.global_state.apples[x][y] = np.random.choice(
                        [0, 1], p=[1 - prob, prob]
                    )

    def _grow_apples_old(self) -> None:
        regrow_matrix = np.zeros(
            (self.global_state.map_height, self.global_state.map_width),
            dtype=np.float32,
        )

        for apple in self.global_state.apples:
            for offset_x, offset_y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                pos_x, pos_y = apple.x + offset_x, apple.y + offset_y
                if self.global_state.in_map(pos_x, pos_y):
                    regrow_matrix[pos_y, pos_x] += self.regrow_chance

        for agent in self.global_state.agent_states.values():
            regrow_matrix[agent.y, agent.x] = 0

        if self.fixed_spawn:
            for x, y in product(
                range(self.global_state.map_width),
                range(self.global_state.map_height),
            ):
                if (x, y) not in self.fixed_spawn_positions:
                    regrow_matrix[y, x] = 0

        for apple in self.global_state.apples:
            regrow_matrix[apple.y, apple.x] = 0
        self.global_state.apples += [
            Apple(x, y)
            for x, y in product(
                range(self.global_state.map_width),
                range(self.global_state.map_height),
            )
            if regrow_matrix[y, x] > np.random.random()
        ]

    def _grow_apples(self) -> None:
        if self.global_state.fixed_spawn:
            self._grow_apples_new()
        else:
            self._grow_apples_old()

    def _clear_rewards(self) -> None:
        self.rewards = {name: 0.0 for name in self.possible_agents}

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],  # observations
        dict[AgentID, float],  # rewards
        dict[AgentID, bool],  # dones
        dict[AgentID, bool],  # truncations
        dict[AgentID, dict],  # infos
    ]:
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        self._clear_rewards()
        self.current_history.append(
            HistoryState(
                board_step=self.global_state.steps_on_board,
                rewards={},
                collected_apples={agent: 0 for agent in self.possible_agents},
                tagged_agents={agent: 0 for agent in self.possible_agents},
                apples_on_board=self.global_state.apples_on_board(),
                actions={},
                agents_pos={
                    a: (s.x, s.y) for a, s in self.global_state.agent_states.items()
                },
            )
        )

        agents_order = self.agents[:]
        random.shuffle(agents_order)
        for agent in agents_order:
            action = actions[agent]

            if self.truncations[agent]:
                continue

            if self.global_state.agent_states[agent].isTagged():
                self.global_state.agent_states[agent].tagged -= 1
                self.current_history[-1].tagged_agents[agent] += 1
            else:
                self.rewards[agent] -= 0.01  # Penalty for every time step
                self._move_reward_agent(agent, Action(action))

            self.current_history[-1].actions[agent] = Action(action)

        self._grow_apples()

        self.global_state.steps_on_board += 1

        env_truncation = self.global_state.steps_on_board >= self.max_cycles

        self.truncations = {agent: env_truncation for agent in self.possible_agents}
        if env_truncation:
            self.agents = []

        self.global_state.cal_obs()

        if self.render_mode != "":
            self.render()

        for a, r in self.rewards.items():
            self.current_history[-1].rewards[a] = r

        self.infos = {
            agent: {
                "neighbours": neighbours,
            }
            for agent, neighbours in self.global_state.get_neighbours().items()
        }

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
            return

        epoch: int = 0
        tag: str = ""
        heatmap: bool = False

        if "epoch" in options:
            epoch = options["epoch"]

        if "tag" in options:
            tag = options["tag"]

        if "heatmap" in options:
            heatmap = options["heatmap"]

        log_name = f"harvest-{tag}"

        cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        collected_apples = {agent: 0.0 for agent in self.possible_agents}
        cumulative_tagged_agents: dict[AgentID, float] = {
            agent: 0.0 for agent in self.possible_agents
        }

        # Scaled to one episode
        scaled_cumulative_rewards: dict[AgentID, float]
        scaled_collected_apples: dict[AgentID, float]

        apples_on_board: float = np.mean(
            np.array([h.apples_on_board for h in self.current_history])
        )

        for history in self.current_history:
            for agent_id in self.possible_agents:
                cumulative_rewards[agent_id] += history.rewards[agent_id]
                collected_apples[agent_id] += history.collected_apples[agent_id]
                cumulative_tagged_agents[agent_id] += history.tagged_agents[agent_id]

        # This is used if multiple episodes are logged at once (e.g. one epoch)
        episodes: int = 1
        if len(self.current_history) > self.max_cycles:
            episodes = len(self.current_history) // self.max_cycles
            if episodes * self.max_cycles != len(self.current_history):
                raise ValueError(
                    "The length of the current history is not a multiple of the max_cycles."
                )
        if episodes != 1:
            scaled_cumulative_rewards = {
                a: r / episodes for a, r in cumulative_rewards.items()
            }
            scaled_collected_apples = {
                a: c / episodes for a, c in collected_apples.items()
            }
        else:
            print("Divider is 1.0 - no scaling of the rewards.")
            scaled_cumulative_rewards = cumulative_rewards
            scaled_collected_apples = collected_apples

        for agent_id in self.possible_agents:
            self.summary_writer.add_scalar(
                f"{log_name}/cumulative_reward/{agent_id}/per_episode",
                scaled_cumulative_rewards[agent_id],
                epoch,
            )

            self.summary_writer.add_scalar(
                f"{log_name}/apples/collected/agents/{agent_id}/per_episode",
                collected_apples[agent_id],
                epoch,
            )

        self.summary_writer.add_scalar(
            f"{log_name}/apples/collected/all/per_episode",
            sum(collected_apples.values()) / self.n_players,
            epoch,
        )

        self.summary_writer.add_scalar(
            f"{log_name}/apples/on_board/per_step",
            apples_on_board,
            epoch,
        )

        # calculate equality
        # e = 1- ((for each i in agents: for each j in agents: abs(rewardI-reawrdJ))/2*n*sum(rewards_aller_agents)))))

        dividend = 0.0
        for i in range(len(self.possible_agents)):
            for j in range(len(self.possible_agents)):
                dividend += abs(
                    scaled_cumulative_rewards[self.possible_agents[i]]
                    - scaled_cumulative_rewards[self.possible_agents[j]]
                )
        divisor = 2.0 * self.n_players * sum(scaled_cumulative_rewards.values())

        e: float = 1.0 - dividend / divisor

        self.summary_writer.add_scalar(
            f"{log_name}/equality",
            e,
            epoch,
        )

        # calculate sustainability sustainability (S) (the average time at which apples are collected)

        # avg_t_per_agent: dict[AgentID, float] = {
        #     agent: 0 for agent in self.possible_agents
        # }
        # for episodes in range(episodes):
        #     episode_avg_t_per_agent: dict[AgentID, float] = {
        #         agent: 0 for agent in self.possible_agents
        #     }
        #     for t in range(self.max_cycles):
        #         for agent in self.possible_agents:
        #             if (
        #                 self.current_history[
        #                     t + episodes * self.max_cycles
        #                 ].collected_apples[agent]
        #                 > 0
        #             ):
        #                 episode_avg_t_per_agent[agent] += t
        #     for agent in self.possible_agents:
        #         avg_t_per_agent[agent] += (
        #             self.max_cycles / episode_avg_t_per_agent[agent]
        #         )
        #
        # for agent in self.possible_agents:
        #     avg_t_per_agent[agent] = avg_t_per_agent[agent] / episodes
        #
        # sustainability: float = sum(avg_t_per_agent.values()) / self.n_players
        # self.summary_writer.add_scalar(
        #     f"{log_name}/sustainability",
        #     sustainability,
        #     epoch,
        # )

        # calculate peace

        peace: float = self.n_players - (
            sum(cumulative_tagged_agents.values()) / len(self.current_history)
        )
        self.summary_writer.add_scalar(
            f"{log_name}/peace",
            peace,
            epoch,
        )

        if heatmap:
            # here x and y are switched so the pictures are horizontal
            all_agents_pos = np.zeros(
                shape=(self.global_state.map_height, self.global_state.map_width)
            )

            agent_pos = {
                agent: np.zeros(
                    shape=(self.global_state.map_height, self.global_state.map_width)
                )
                for agent in self.possible_agents
            }
            agent_divider: float = 1.0 / len(self.possible_agents)

            for history in self.current_history:
                for agent_id, pos in history.agents_pos.items():
                    agent_pos[agent_id][pos[1], pos[0]] += 1
                    all_agents_pos[pos[1], pos[0]] += agent_divider

            all_agents_pos /= len(self.current_history)
            image_data = torch.from_numpy(all_agents_pos).unsqueeze(0).unsqueeze(0)

            # Log the 2D position as an image
            self.summary_writer.add_image(
                f"{log_name}/heatmap/all_agents",
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
    # env = env(render_mode="human")
    # api_test(env, num_cycles=1000, verbose_progress=True)
    #
    from pettingzoo.test.parallel_test import parallel_api_test

    parallel_api_test(
        parallel_env(fixed_spawn=True, grid_width=25, grid_height=9), num_cycles=1000
    )
