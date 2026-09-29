import numpy as np
import matplotlib.pyplot as plt


# DATA
x_data = np.array([0, 2, 1, 3], dtype=float)
y_data = np.array([0.5, 3.5, 1.5, 7.5], dtype=float)


# GENERIC MSE

def mse_line(m, b, x, y):
    y_pred = m * x + b
    return np.mean((y - y_pred) ** 2)


def mse_parabola(a, b, c, x, y):
    y_pred = a * x**2 + b * x + c
    return np.mean((y - y_pred) ** 2)


# precompute the power sums used throughout, similar to hand derivation

def power_sums(x, y):
    n = len(x)
    return {
        "n": n,
        "Sx": np.sum(x),
        "Sy": np.sum(y),
        "Sxx": np.sum(x**2),
        "Sxy": np.sum(x * y),
        "Sxxx": np.sum(x**3),
        "Sxxxx": np.sum(x**4),
        "Sxxy": np.sum(x**2 * y),
    }


# PART A: LINE FIT  y = m*x + b

def line_analytical(x, y):
    # closed-form least-squares solution from normal equations
    s = power_sums(x, y)
    n, Sx, Sy, Sxx, Sxy = s["n"], s["Sx"], s["Sy"], s["Sxx"], s["Sxy"]

    # solve:
    #   [Sxx  Sx ] [m]   [Sxy]
    #   [Sx   n  ] [b] = [Sy ]
    A = np.array([[Sxx, Sx],
                  [Sx,  n]])
    rhs = np.array([Sxy, Sy])
    m, b = np.linalg.solve(A, rhs)
    return m, b


def line_newton_alternating(x, y, m0=0.0, b0=0.0, iterations=15):
    # fix b, solve exactly for m, then fix m and solve exactly for b
    s = power_sums(x, y)
    n, Sx, Sy, Sxx, Sxy = s["n"], s["Sx"], s["Sy"], s["Sxx"], s["Sxy"]

    m, b = m0, b0
    history = [(m, b, mse_line(m, b, x, y))]

    for _ in range(iterations):
        # fix b, solve dMSE/dm = 0 for m:  Sxy = m*Sxx + b*Sx
        m = (Sxy - b * Sx) / Sxx
        # fix (new) m, solve dMSE/db = 0 for b:  Sy = m*Sx + b*n
        b = (Sy - m * Sx) / n
        history.append((m, b, mse_line(m, b, x, y)))

    return m, b, history


# PART B: PARABOLA FIT  y = a*x^2 + b*x + c

def parabola_analytical(x, y):
    # closed-form least-squares solutions from 3 normal equations
    s = power_sums(x, y)
    n = s["n"]
    Sx, Sxx, Sxxx, Sxxxx = s["Sx"], s["Sxx"], s["Sxxx"], s["Sxxxx"]
    Sy, Sxy, Sxxy = s["Sy"], s["Sxy"], s["Sxxy"]

    A = np.array([
        [Sxxxx, Sxxx, Sxx],
        [Sxxx,  Sxx,  Sx],
        [Sxx,   Sx,   n],
    ])
    rhs = np.array([Sxxy, Sxy, Sy])
    a, b, c = np.linalg.solve(A, rhs)
    return a, b, c


def parabola_newton_alternating(x, y, a0=0.0, b0=0.0, c0=0.0, iterations=25):
    # cycle a, b, c, each solved exactly + hold other 2 fixed
    s = power_sums(x, y)
    n = s["n"]
    Sx, Sxx, Sxxx, Sxxxx = s["Sx"], s["Sxx"], s["Sxxx"], s["Sxxxx"]
    Sy, Sxy, Sxxy = s["Sy"], s["Sxy"], s["Sxxy"]

    a, b, c = a0, b0, c0
    history = [(a, b, c, mse_parabola(a, b, c, x, y))]

    for _ in range(iterations):
        # fix b, c -> solve for a:  Sxxy = a*Sxxxx + b*Sxxx + c*Sxx
        a = (Sxxy - b * Sxxx - c * Sxx) / Sxxxx
        # fix a (new), c -> solve for b:  Sxy = a*Sxxx + b*Sxx + c*Sx
        b = (Sxy - a * Sxxx - c * Sx) / Sxx
        # fix a, b (new) -> solve for c:  Sy = a*Sxx + b*Sx + c*n
        c = (Sy - a * Sxx - b * Sx) / n
        history.append((a, b, c, mse_parabola(a, b, c, x, y)))

    return a, b, c, history


# PLOTS

def plot_fit(x, y, line_params, parabola_params, title="Least-Squares Fits"):
    m, b = line_params
    a, bb, c = parabola_params

    x_smooth = np.linspace(min(x) - 1, max(x) + 1, 200)
    y_line = m * x_smooth + b
    y_parab = a * x_smooth**2 + bb * x_smooth + c

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="red", s=80, zorder=5, label="Data points")
    plt.plot(x_smooth, y_line, color="green", linewidth=2,
              label=f"Line fit: y={m:.4f}x+{b:.4f}")
    plt.plot(x_smooth, y_parab, color="blue", linewidth=2,
              label=f"Parabola fit: y={a:.4f}x^2+{bb:.4f}x+{c:.4f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_part2_fits.png", dpi=120)
    plt.show()


def plot_convergence(line_history, parabola_history):
    line_mse = [h[2] for h in line_history]
    parab_mse = [h[3] for h in parabola_history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(range(len(line_mse)), line_mse, "o-", color="green")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Line fit: alternating Newton-Raphson convergence")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(len(parab_mse)), parab_mse, "o-", color="blue")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Parabola fit: alternating Newton-Raphson convergence")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot_part2_convergence.png", dpi=120)
    plt.show()


def plot_fit_progression(x, y, history, model, snapshot_iters, title, curve_label_fmt):
    x_smooth = np.linspace(min(x) - 1, max(x) + 1, 200)

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, color="red", s=80, zorder=5, label="Data points")

    n_snaps = len(snapshot_iters)
    for i, it in enumerate(snapshot_iters):
        it = min(it, len(history) - 1)
        *params, mse_val = history[it]
        y_smooth = model(params, x_smooth)

        is_last = (i == n_snaps - 1)
        alpha = 0.35 + 0.65 * (i / max(n_snaps - 1, 1))
        linewidth = 2.8 if is_last else 1.2

        label = f"iter {it}: {curve_label_fmt(params)}"
        if is_last:
            label += "  (converged)"

        plt.plot(x_smooth, y_smooth, color="blue", alpha=alpha,
                  linewidth=linewidth, label=label)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    safe_title = title.replace(" ", "_").replace(":", "")
    plt.savefig(f"plot_{safe_title}.png", dpi=120)
    plt.show()


def plot_parameter_trajectory_line(history):
    m_vals = [h[0] for h in history]
    b_vals = [h[1] for h in history]

    plt.figure(figsize=(6, 6))
    plt.plot(m_vals, b_vals, "o-", color="darkgreen")
    plt.scatter([m_vals[0]], [b_vals[0]], color="red", s=100, zorder=5, label="Start")
    plt.scatter([m_vals[-1]], [b_vals[-1]], color="purple", s=100, zorder=5, label="Converged")
    plt.xlabel("m")
    plt.ylabel("b")
    plt.title("Line fit: (m, b) trajectory across iterations")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_part2_line_trajectory.png", dpi=120)
    plt.show()


# main driver

if __name__ == "__main__":

    print("=" * 60)
    print("Preliminary power sums")
    s = power_sums(x_data, y_data)
    for k, v in s.items():
        print(f"  {k} = {v}")

    # line fit
    print("\n" + "=" * 60)
    print("LINE FIT  y = m*x + b")

    m_analytical, b_analytical = line_analytical(x_data, y_data)
    print(f"\nAnalytical solution: m = {m_analytical:.6f}, b = {b_analytical:.6f}")
    print(f"MSE (analytical)   : {mse_line(m_analytical, b_analytical, x_data, y_data):.6f}")

    m_newton, b_newton, line_history = line_newton_alternating(x_data, y_data, iterations=15)
    print(f"\nNewton-Raphson (alternating), 15 iterations:")
    print(f"  m = {m_newton:.6f}, b = {b_newton:.6f}")
    print(f"  MSE = {mse_line(m_newton, b_newton, x_data, y_data):.6f}")
    print(f"  |m_analytical - m_newton| = {abs(m_analytical - m_newton):.2e}")
    print(f"  |b_analytical - b_newton| = {abs(b_analytical - b_newton):.2e}")

    print("\nIteration-by-iteration (m, b, MSE):")
    for i, (m_i, b_i, mse_i) in enumerate(line_history):
        print(f"  iter {i}: m={m_i:.6f}, b={b_i:.6f}, MSE={mse_i:.6f}")

    # parabola fit
    print("\n" + "=" * 60)
    print("PARABOLA FIT  y = a*x^2 + b*x + c")

    a_analytical, b2_analytical, c_analytical = parabola_analytical(x_data, y_data)
    print(f"\nAnalytical solution: a = {a_analytical:.6f}, b = {b2_analytical:.6f}, c = {c_analytical:.6f}")
    print(f"MSE (analytical)   : {mse_parabola(a_analytical, b2_analytical, c_analytical, x_data, y_data):.6f}")

    a_newton, b2_newton, c_newton, parab_history = parabola_newton_alternating(
        x_data, y_data, iterations=25
    )
    print(f"\nNewton-Raphson (alternating), 25 iterations:")
    print(f"  a = {a_newton:.6f}, b = {b2_newton:.6f}, c = {c_newton:.6f}")
    print(f"  MSE = {mse_parabola(a_newton, b2_newton, c_newton, x_data, y_data):.6f}")
    print(f"  |a_analytical - a_newton| = {abs(a_analytical - a_newton):.2e}")
    print(f"  |b_analytical - b_newton| = {abs(b2_analytical - b2_newton):.2e}")
    print(f"  |c_analytical - c_newton| = {abs(c_analytical - c_newton):.2e}")

    print("\nIteration-by-iteration (a, b, c, MSE) [first 10 shown]:")
    for i, (a_i, b_i, c_i, mse_i) in enumerate(parab_history[:10]):
        print(f"  iter {i}: a={a_i:.6f}, b={b_i:.6f}, c={c_i:.6f}, MSE={mse_i:.6f}")

    # plots
    plot_fit(x_data, y_data,
             (m_analytical, b_analytical),
             (a_analytical, b2_analytical, c_analytical),
             title="Part 2: Least-Squares Line and Parabola Fits")

    plot_convergence(line_history, parab_history)
    plot_parameter_trajectory_line(line_history)

    # fit-progression plots
    plot_fit_progression(
        x_data, y_data, line_history,
        model=lambda params, xs: params[0] * xs + params[1],
        snapshot_iters=[0, 1, 2, 5, len(line_history) - 1],
        title="Part 2: Line fit progression",
        curve_label_fmt=lambda p: f"y={p[0]:.3f}x+{p[1]:.3f}",
    )

    plot_fit_progression(
        x_data, y_data, parab_history,
        model=lambda params, xs: params[0] * xs**2 + params[1] * xs + params[2],
        snapshot_iters=[0, 1, 3, 8, len(parab_history) - 1],
        title="Part 2: Parabola fit progression",
        curve_label_fmt=lambda p: f"y={p[0]:.3f}x^2+{p[1]:.3f}x+{p[2]:.3f}",
    )