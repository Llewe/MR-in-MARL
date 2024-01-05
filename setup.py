from setuptools import Extension, setup
from Cython.Build import cythonize


# setup(ext_modules=cythonize("src/envs/aec/my_coin_game_cy.pyx"))
# setup(ext_modules=cythonize("src/envs/aec/harvest_cy.pyx"))

extensions = [
    Extension("*", ["src/*.pyx"], include_dirs=["src/*"]),
    Extension("*", ["src/*.py"]),
]

setup(
    ext_modules=cythonize(
        extensions,
        build_dir="build",
    ),
)
