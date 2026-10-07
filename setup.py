"""Build the collision kernel without a NumPy build dependency."""

import sys

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "contact_solver.solvers._cubic_minima",
            ["src/contact_solver/solvers/_cubic_minima.c"],
            extra_compile_args=(
                ["/O2", "/fp:precise"] if sys.platform == "win32" else ["-O3", "-fno-fast-math"]
            ),
            optional=True,
        )
    ]
)
