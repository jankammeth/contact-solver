"""Test solver infrastructure."""

import numpy as np
import pytest

from contact_solver.solvers import Solver, Constraint
from contact_solver.solvers.constraints import build_dynamics_constraints
from contact_solver.solvers.cubic_utils import cubic_coefficients, eval_cubic, eval_cubic_vel
from contact_solver.solvers.osqp_utils import OSQPSettings


def test_constraint_dataclass():
    import scipy.sparse as sp
    
    matrix = sp.eye(3, format="csc")
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])
    
    c = Constraint(matrix, lower, upper)
    assert c.matrix.shape == (3, 3)
    assert len(c.lower) == 3
    assert len(c.upper) == 3


def test_osqp_settings_defaults():
    settings = OSQPSettings()
    assert settings.max_iter == 4000
    assert settings.polish is True


def test_cubic_coefficients():
    p0 = np.array([0.0, 0.0])
    v0 = np.array([0.0, 0.0])
    pf = np.array([1.0, 0.0])
    vf = np.array([0.0, 0.0])
    T = 1.0
    
    a, b, c, d = cubic_coefficients(p0, v0, pf, vf, T)
    
    # Check endpoints
    pos_0 = eval_cubic((a, b, c, d), 0.0)
    pos_T = eval_cubic((a, b, c, d), T)
    
    np.testing.assert_allclose(pos_0, p0, atol=1e-10)
    np.testing.assert_allclose(pos_T, pf, atol=1e-10)
    
    # Check velocities
    vel_0 = eval_cubic_vel((a, b, c, d), 0.0)
    vel_T = eval_cubic_vel((a, b, c, d), T)
    
    np.testing.assert_allclose(vel_0, v0, atol=1e-10)
    np.testing.assert_allclose(vel_T, vf, atol=1e-10)


def test_build_dynamics_constraints():
    K_i = [10, 10]  # 2 robots, 10 timesteps each
    h = 0.1
    
    initial_states = {
        "position": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "velocity": np.array([[0.0, 0.0], [0.0, 0.0]]),
    }
    final_states = {
        "position": np.array([[1.0, 1.0], [0.0, 1.0]]),
        "velocity": np.array([[0.0, 0.0], [0.0, 0.0]]),
    }
    box_limits = {
        "jerk_min": -20.0,
        "jerk_max": 20.0,
        "acc_min": -10.0,
        "acc_max": 10.0,
        "vel_min": -2.0,
        "vel_max": 2.0,
        "pos_min": np.array([-10.0, -10.0]),
        "pos_max": np.array([10.0, 10.0]),
    }
    
    constraints = build_dynamics_constraints(
        K_i, h, initial_states, final_states, box_limits
    )
    
    assert "jerk" in constraints
    assert "acceleration" in constraints
    assert "velocity" in constraints
    assert "position" in constraints
    
    # Check dimensions: 2 robots * 10 timesteps * 2 dims = 40 decision vars
    total_vars = sum(K * 2 for K in K_i)  # acceleration per timestep per dim
    assert constraints["acceleration"].matrix.shape[1] == total_vars
