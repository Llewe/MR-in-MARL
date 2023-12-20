from Cython.Build import cythonize
from setuptools import setup

# setup(ext_modules=cythonize("src/envs/aec/my_coin_game_cy.pyx"))
# setup(ext_modules=cythonize("src/envs/aec/harvest_cy.pyx"))
setup(
    ext_modules=cythonize(
        [
            "src/cfg_manager.pyx",
            "src/training.pyx",
            "src/agents/utils/network.pyx",
            "src/agents/utils/reward_normalization.pyx",
            "src/agents/utils/state_value_network.pyx",
            "src/agents/a2c.pyx",
            "src/agents/mate.pyx",
            "src/agents/random_agents.pyx",
            "src/agents/random_agents.pyx",
            "src/envs/aec/harvest.pyx",
            "src/envs/aec/my_coin_game.pyx",
        ]
    )
)
