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
            "src/agents/env_specific/coingame/ma_coin_to_middle.pyx",
            "src/agents/env_specific/ipd/punish_defect.pyx",
            "src/agents/a2c.pyx",
            "src/agents/gifting.pyx",
            #     "src/agents/lola_pg.pyx",
            "src/agents/mate.pyx",
            "src/agents/mr_agent_a2c.pyx",
            "src/agents/random_agents.pyx",
            "src/agents/random_agents.pyx",
            "src/envs/aec/harvest.pyx",
            "src/envs/aec/dilemma/dilemma_pettingzoo.pyx",
            "src/envs/aec/dilemma/games.pyx",
            "src/envs/aec/my_coin_game.pyx",
            "src/utils/loggers/obs_logger.pyx",
            "src/utils/loggers/simple_env_logger.pyx",
            "src/utils/loggers/util_logger.pyx",
            "src/utils/gym_utils.pyx",
        ]
    )
)
