"""Kinematic bicycle model (report sections 2-3).

State  x = [x, y, theta]  : position of the rear-axle midpoint P and heading
Input  u = [v, delta]     : speed of P and front-wheel steering angle
"""
import numpy as np

# Parameters (starting values, see report)
L = 0.3                       # wheelbase [m]
V_MAX = 1.0                   # |v| <= V_MAX [m/s]
DELTA_MAX = np.deg2rad(30)    # |delta| <= DELTA_MAX [rad]
DT = 0.1                      # time step [s]
R_MIN = L / np.tan(DELTA_MAX) # minimum turning radius [m]


def f(x, u, L=L):
    """Continuous-time model x_dot = f(x, u)  (eq. 6)."""
    _, _, th = x
    v, d = u
    return np.array([v * np.cos(th),
                     v * np.sin(th),
                     v / L * np.tan(d)])


def step(x, u, dt=DT, L=L):
    """Discrete-time model x_{k+1} = x_k + f(x_k, u_k) dt  (forward Euler, eq. 7)."""
    return np.asarray(x, dtype=float) + dt * f(x, u, L)


def linearize(x_bar, u_bar, dt=DT, L=L):
    """Linearise the discrete model around (x_bar, u_bar):
        x_{k+1} ~ A x_k + B u_k + c      (Jacobians: Appendix B, eq. 14)
    """
    th = x_bar[2]
    v, d = u_bar
    A = np.array([[1.0, 0.0, -v * np.sin(th) * dt],
                  [0.0, 1.0,  v * np.cos(th) * dt],
                  [0.0, 0.0,  1.0]])
    B = np.array([[np.cos(th) * dt, 0.0],
                  [np.sin(th) * dt, 0.0],
                  [np.tan(d) / L * dt, v * dt / (L * np.cos(d) ** 2)]])
    c = step(x_bar, u_bar, dt, L) - A @ x_bar - B @ u_bar
    return A, B, c


if __name__ == "__main__":
    # Test 1: constant v and delta should drive a circle of radius L / tan(delta)
    import matplotlib.pyplot as plt

    v, d = 0.5, np.deg2rad(20)
    R_expected = L / np.tan(d)
    T = 2 * np.pi * R_expected / v              # time for one full circle
    n = int(round(T / DT))

    xs = [np.zeros(3)]
    for _ in range(n):
        xs.append(step(xs[-1], [v, d]))
    xs = np.array(xs)

    # fit a circle through the simulated points (least squares: x^2 + y^2 = 2ax + 2by + c)
    P = xs[:, :2]
    M = np.column_stack([2 * P, np.ones(len(P))])
    a, b, cc = np.linalg.lstsq(M, (P ** 2).sum(axis=1), rcond=None)[0]
    center = np.array([a, b])
    R_sim = np.sqrt(cc + a ** 2 + b ** 2)
    print(f"expected radius  : {R_expected:.4f} m")
    print(f"simulated radius : {R_sim:.4f} m")

    # Test 2: the linearisation should match the true step for small deviations
    x_bar, u_bar = np.array([1.0, 2.0, 0.7]), np.array([0.6, 0.2])
    A, B, c = linearize(x_bar, u_bar)
    dx, du = np.array([0.01, -0.01, 0.02]), np.array([0.02, -0.01])
    err = np.linalg.norm(step(x_bar + dx, u_bar + du) - (A @ (x_bar + dx) + B @ (u_bar + du) + c))
    print(f"linearisation error for a small deviation: {err:.1e}  (should be tiny)")

    ang = np.linspace(0, 2 * np.pi, 200)
    plt.plot(center[0] + R_expected * np.cos(ang), center[1] + R_expected * np.sin(ang),
             "k--", label="expected circle")
    plt.plot(xs[:, 0], xs[:, 1], "b-", label="simulated (Euler)")
    plt.axis("equal"); plt.grid(True); plt.legend()
    plt.title(f"Circle test: v = {v} m/s, delta = 20 deg")
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.show()
