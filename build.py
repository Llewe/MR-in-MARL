import os


import os
import shutil
from distutils.core import Distribution, Extension

from Cython.Build import build_ext, cythonize


def find_cython_files():
    cython_files = []
    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py") and not file.endswith("__init__.py"):
                cython_files.append(os.path.join(root, file))
    return cython_files


cython_dir = os.path.join("build", "cython")
extension = Extension(
    "name",
    find_cython_files(),
)

ext_modules = cythonize([extension], include_path=["src"], build_dir=cython_dir)
dist = Distribution({"ext_modules": ext_modules})
cmd = build_ext(dist)
cmd.ensure_finalized()
cmd.run()

for output in cmd.get_outputs():
    relative_extension = os.path.relpath(output, cmd.build_lib)
    shutil.copyfile(output, relative_extension)
