"""MPC for the kinematic bicycle, WITHOUT obstacles (stage 1, step 3).

Follows Algorithm 1 of the preliminary report (Appendix B), lines 7-28,
without the obstacle constraints (lines 19-20) and the position box (line 21).

At every step the nonlinear model is linearised around a guess trajectory
(x_bar, u_bar), so the MPC problem is a QP (quadratic cost, linear constraints).
One addition to Algorithm 1: a trust region on the steering (see TRUST_DELTA).
No ROS imports here: the same file is used offline and in Gazebo.
"""
import numpy as np
import cvxpy as cp
from model import step, linearize, V_MAX, DELTA_MAX

N = 20                      # prediction horizon [steps] (N * DT = 2 s ahead)

# Weights of the cost (report eq. 9, Algorithm 1 line 15)
Q = np.diag([1.0, 1.0])     # position error: x and y are equally important
R = np.diag([0.1, 0.1])     # input effort: small, so it only smooths the inputs and does not slow the car much
P = 10 * Q                  # terminal cost: the end of the plan should be close to B (pulls the plan towards the goal)

# Trust region (NOT in Algorithm 1): the linearised model is only accurate close to the guess.
# Without this, the plan jumped between -30 and +30 deg steering and its predictions were up to 4 m off
# the true model. With |delta_k - delta_bar_k| <= 5 deg the error stays below ~0.8 m.
TRUST_DELTA = np.deg2rad(5)


def initial_guess(x0):
    """Algorithm 1 lines 8-9: drive straight at half speed, rolled out with the nonlinear model.
    Returns x_bar (3 x N+1) and u_bar (2 x N)."""
    u_bar = np.tile([[V_MAX / 2], [0.0]], (1, N))     # u_k = (V_MAX/2, 0) for all k
    x_bar = np.zeros((3, N + 1))
    x_bar[:, 0] = x0
    for k in range(N):
        x_bar[:, k + 1] = step(x_bar[:, k], u_bar[:, k])
    return x_bar, u_bar


def solve_mpc(x0, goal, x_bar, u_bar):
    """Solve the MPC problem once (Algorithm 1 lines 11-21, without obstacles).

    x0     : current state [x, y, theta]
    goal   : goal position B = [x_B, y_B]
    x_bar, u_bar : guess trajectory to linearise around (3 x N+1, 2 x N)
    Returns the optimal inputs u (2 x N) and predicted states x (3 x N+1).
    """
    x = cp.Variable((3, N + 1))     # x_0 ... x_N
    u = cp.Variable((2, N))         # u_0 ... u_{N-1}

    cost = 0
    constraints = [x[:, 0] == x0]                                  # line 16: x_0 = x(t)
    for k in range(N):
        A, B, c = linearize(x_bar[:, k], u_bar[:, k])               # lines 11-13: A_k, B_k, c_k
        constraints += [x[:, k + 1] == A @ x[:, k] + B @ u[:, k] + c]   # line 17: linearised model
        constraints += [cp.abs(u[0, k]) <= V_MAX,                   # line 18: |v_k| <= v_max
                        cp.abs(u[1, k]) <= DELTA_MAX]               #          |delta_k| <= delta_max
        constraints += [cp.abs(u[1, k] - u_bar[1, k]) <= TRUST_DELTA]   # trust region: stay near the guess
        p_k = x[:2, k]                                              # position part of the state
        cost += cp.quad_form(p_k - goal, Q) + cp.quad_form(u[:, k], R)  # line 15: stage cost
    cost += cp.quad_form(x[:2, N] - goal, P)                        # line 15: terminal cost

    problem = cp.Problem(cp.Minimize(cost), constraints)
    # QP, so any QP solver works. We name Clarabel explicitly: otherwise CVXPY picks Gurobi (installed),
    # whose free pip licence refuses quadratic problems with more than 200 variables (ours has ~350 after CVXPY's reformulation).
    problem.solve(solver=cp.CLARABEL)
    if problem.status != cp.OPTIMAL:
        return None, None                                           # line 22: infeasible / failed
    return u.value, x.value


def shift(x_opt, u_opt):
    """Algorithm 1 lines 26-28: warm start for the next step.
    Drop the first step, repeat the last input, and extend the states with one model step."""
    u_bar = np.hstack([u_opt[:, 1:], u_opt[:, -1:]])                       # [u*_1 ... u*_{N-1}, u*_{N-1}]
    x_bar = np.hstack([x_opt[:, 1:], step(x_opt[:, -1], u_opt[:, -1])[:, None]])  # [x*_1 ... x*_N, f(x*_N, u*_{N-1})]
    return x_bar, u_bar
