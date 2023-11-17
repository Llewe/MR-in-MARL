from gymnasium import Space
from gymnasium.spaces import Box, Discrete


def get_space_size(space: Space) -> int:
    if isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Box):
        return space.shape[0]
    else:
        raise ValueError("Unsupported space type")
