"""OSQP solver utilities."""

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

from .constraints import Constraint


@dataclass
class OSQPSettings:
    max_iter: int = 4000
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5
    polish: bool = True
    warm_start: bool = True
    verbose: bool = False

    @classmethod
    def from_config(cls, config) -> "OSQPSettings":
        return cls(
            max_iter=config.solver.osqp_max_iter,
            eps_abs=config.solver.osqp_eps_abs,
            eps_rel=config.solver.osqp_eps_rel,
            polish=config.solver.osqp_polish,
            warm_start=config.solver.osqp_warm_start,
            verbose=config.solver.osqp_verbose,
        )


def _is_osqp_1x() -> bool:
    version = getattr(osqp, "__version__", "0.0.0")
    return version.startswith("1.")


def solve_qp(
    P: sp.csc_matrix,
    q: np.ndarray,
    constraints: list[Constraint],
    warm_start: np.ndarray | None = None,
    settings: OSQPSettings | None = None,
) -> tuple[np.ndarray, dict]:
    settings = settings or OSQPSettings()

    A = sp.vstack([c.matrix for c in constraints], format="csc")
    lower = np.concatenate([c.lower for c in constraints])
    u = np.concatenate([c.upper for c in constraints])

    is_1x = _is_osqp_1x()
    setup_kwargs = {
        "P": P,
        "q": q,
        "A": A,
        "l": lower,
        "u": u,
        "verbose": settings.verbose,
        "max_iter": settings.max_iter,
        "eps_abs": settings.eps_abs,
        "eps_rel": settings.eps_rel,
    }

    if is_1x:
        setup_kwargs["warm_starting"] = settings.warm_start
        setup_kwargs["polishing"] = settings.polish
    else:
        setup_kwargs["warm_start"] = settings.warm_start
        setup_kwargs["polish"] = settings.polish

    problem = osqp.OSQP()
    problem.setup(**setup_kwargs)

    if warm_start is not None:
        problem.warm_start(x=warm_start)

    result = problem.solve()
    info = result.info

    prim_res = getattr(info, "prim_res", None) or getattr(info, "pri_res", None)
    dual_res = getattr(info, "dual_res", None) or getattr(info, "dua_res", None)

    metrics = {
        "status": info.status,
        "status_val": info.status_val,
        "obj_val": info.obj_val,
        "iter": info.iter,
        "pri_res": prim_res,
        "dua_res": dual_res,
        "setup_time": info.setup_time,
        "solve_time": info.solve_time,
        "polish_time": getattr(info, "polish_time", 0.0),
        "run_time": info.run_time,
    }

    converged = info.status_val in (1, 2)

    if not converged:
        raise RuntimeError(
            f"OSQP failed: status='{info.status}', code={info.status_val}, "
            f"iter={info.iter}, prim_res={prim_res}, dual_res={dual_res}"
        )

    return result.x, metrics
