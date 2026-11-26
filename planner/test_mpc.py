"""Offline closed-loop test of mpc.py (no Gazebo, no ROS).

The "real car" is the nonlinear model step(). Receding horizon (MPC I p31):
solve the MPC, apply only u_0, move the car one step, re-plan.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from model import step, DT
from mpc import initial_guess, solve_mpc, shift

GOAL_RADIUS = 0.2       # success when ||p - B|| <= 0.2 m
T_MAX = 30.0            # time limit [s]


def run(start, goal):
    """Closed loop from start (x, y, theta) to goal (x, y). Returns driven states, plans, solve times."""
    x = np.array(start, dtype=float)
    goal = np.array(goal, dtype=float)
    x_bar, u_bar = initial_guess(x)
    states, plans, solve_times = [x.copy()], [], []

    for t in range(int(T_MAX / DT)):
        t0 = time.perf_counter()
        u_opt, x_opt = solve_mpc(x, goal, x_bar, u_bar)
        solve_times.append(time.perf_counter() - t0)
        if u_opt is None:
            print(f"  step {t}: MPC failed, stop")
            break
        plans.append(x_opt)

        x = step(x, u_opt[:, 0])                 # apply ONLY the first input to the "real car"
        states.append(x.copy())
        x_bar, u_bar = shift(x_opt, u_opt)       # warm start for the next step

        if np.linalg.norm(x[:2] - goal) <= GOAL_RADIUS:
            break
    return np.array(states), plans, solve_times


def report_and_plot(name, start, goal):
    states, plans, solve_times = run(start, goal)
    reached = np.linalg.norm(states[-1, :2] - np.array(goal)) <= GOAL_RADIUS
    steps = len(states) - 1
    print(f"{name}: start {start} -> goal {goal}")
    print(f"  reached goal       : {reached}")
    print(f"  number of steps    : {steps}")
    print(f"  time taken         : {steps * DT:.1f} s (simulated)")
    print(f"  avg solve time/step: {1000 * np.mean(solve_times):.1f} ms (incl. building the problem)")
    print(f"  final state        : x = {states[-1, 0]:.2f}, y = {states[-1, 1]:.2f}, "
          f"theta = {np.degrees(states[-1, 2]):.0f} deg")

    plt.figure()
    for i in range(0, len(plans), 10):           # predicted horizon every 10th step
        plt.plot(plans[i][0], plans[i][1], "--", color="orange", alpha=0.8,
                 label="predicted horizon (every 10th step)" if i == 0 else None)
        plt.plot(plans[i][0, 0], plans[i][1, 0], ".", color="orange")
    plt.plot(states[:, 0], states[:, 1], "b-", label="driven path of P")
    plt.plot(*start[:2], "go", markersize=10, label="start A")
    plt.plot(*goal, "r*", markersize=15, label="goal B")
    plt.axis("equal"); plt.grid(True); plt.legend()
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title(f"{name}: MPC without obstacles")


if __name__ == "__main__":
    report_and_plot("Test 1", (0.0, 0.0, 0.0), (6.0, 0.0))
    report_and_plot("Test 2", (0.0, 0.0, 0.0), (2.0, 3.0))
    plt.show()
