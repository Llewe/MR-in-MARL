from typing import Union

import numpy
from gymnasium import Space
from gymnasium.spaces import Box, Discrete

from gymnasium.spaces.utils import flatdim


def get_space_size(space: Space) -> Union[numpy.int64, int]:
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        return flatdim(space)  # old was space.shape[0]
    else:
        raise ValueError("Unsupported space type")


def add_action_to_space(space: Space, actions: int) -> Space:
    if isinstance(space, Discrete):
        return Discrete(space.n + actions)
    elif isinstance(space, Box):
        return Box(
            low=space.low,
            high=space.high,
            shape=(space.shape[0] + actions,),
            dtype=space.dtype,
        )
    else:
        raise ValueError("Unsupported space type")


def get_action_from_index(gym_space, index):
    if isinstance(gym_space, Discrete):
        num_actions = gym_space.n

        # Handle negative index
        if index < 0:
            index = (
                num_actions + index
            )  # Convert negative index to positive index from the end

        # Ensure index is within valid range
        index = max(0, min(num_actions - 1, index))

        # Retrieve the action from the gym space
        selected_action = index

    else:
        raise ValueError("Unsupported gym space type")

    return selected_action
