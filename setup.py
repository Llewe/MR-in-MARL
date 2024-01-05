from setuptools import setup, find_packages
from Cython.Build import cythonize
import os

from Cython.Distutils import build_ext

# setup(ext_modules=cythonize("src/envs/aec/my_coin_game_cy.pyx"))
# setup(ext_modules=cythonize("src/envs/aec/harvest_cy.pyx"))


# Function to find all Cython files in the /src directory and its subdirectories
def find_cython_files():
    cython_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                cython_files.append(os.path.join(root, file))
    return cython_files


# Use cythonize to build Cython extensions
extensions = cythonize(
    find_cython_files(),
    build_dir="build",
    compiler_directives={
        "language_level": "3",
        "always_allow_keywords": True,
    },
)

setup(
    name="myapp",
    version="1.0.0",
    ext_modules=extensions,
    cmdclass=dict(build_ext=build_ext),
    packages=["src"],
)
