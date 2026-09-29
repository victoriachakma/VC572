import numpy as np
import matplotlib.pyplot as plt


# distance from a point to a function

# for point p = (px, py) and function y = f(x), the squared
# distance from p to a point (x, f(x)) on the curve is:
# D^2(x) = (x - px)^2 + (f(x) - py)^2

def squared_distance(x, px, py, f):
    # return squared distance from (px, py) to (x, f(x))
    return (x - px)**2 + (f(x) - py)**2

def distance(x, px, py, f):
    # return distance from (px, py) to (x, f(x))
    return np.sqrt(squared_distance(x, px, py, f))


# numerical derivatives (fallback for when df/ddf aren't supplied)

def numerical_derivative(g, x, h=1e-5):
    # central-difference approximation of the first derivative
    return (g(x + h) - g(x - h)) / (2 * h)

def numerical_second_derivative(g, x, h=1e-4):
    # central-difference approximation of the second derivative
    return (g(x + h) - 2 * g(x) + g(x - h)) / h**2


# METHOD 1: Newton-Raphson
# finds stationary point of squared-distance function

def newton_raphson(px, py, f, x0, df=None, ddf=None,
                    tolerance=1e-10, max_iterations=100):
    fprime = df if df is not None else (lambda x: numerical_derivative(f, x))
    fdprime = ddf if ddf is not None else (lambda x: numerical_second_derivative(f, x))

    x = x0
    history = [x]
    for i in range(max_iterations):
        g_prime = 2 * (x - px) + 2 * (f(x) - py) * fprime(x)
        g_double_prime = 2 + 2 * fprime(x)**2 + 2 * (f(x) - py) * fdprime(x)

        if abs(g_double_prime) < 1e-14:
            break

        x_new = x - g_prime / g_double_prime
        history.append(x_new)

        if abs(x_new - x) < tolerance:
            x = x_new
            break

        x = x_new

    return x, distance(x, px, py, f), history


# METHOD 2: Golden Section Search
# directly minimizes squared-distance function over specified interval,
# no derivatives required

def golden_section_search(px, py, f, a, b,
                           tolerance=1e-10,
                           max_iterations=1000):

    g = lambda x: squared_distance(x, px, py, f)

    golden_ratio = (np.sqrt(5) - 1) / 2  # == 1 - 1/phi == 2 - phi

    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    history = [c, d]
    iterations = 0

    for i in range(max_iterations):
        iterations += 1

        if g(c) < g(d):
            b = d
            d = c
            c = b - golden_ratio * (b - a)
        else:
            a = c
            c = d
            d = a + golden_ratio * (b - a)

        history.extend([c, d])

        if abs(b - a) < tolerance:
            break

    x_min = (a + b) / 2

    return x_min, distance(x_min, px, py, f), history, iterations


# plots

def plot_search(px, py, f, newton_history, golden_history,
                 x_min, title, x_range=(-10, 10)):

    x_values = np.linspace(x_range[0], x_range[1], 1000)
    y_values = f(x_values)

    plt.figure(figsize=(10, 6))

    plt.plot(x_values, y_values,
              label="Function f(x)", color="blue", linewidth=2)

    plt.scatter(px, py,
                color="red", s=80, zorder=5,
                label=f"Point ({px}, {py})")

    newton_y = [f(x) for x in newton_history]
    plt.plot(newton_history, newton_y,
              "o--", color="green",
              label="Newton-Raphson steps",
              alpha=0.8)

    golden_y = [f(x) for x in golden_history]
    plt.plot(golden_history, golden_y,
              "x--", color="orange",
              label="Golden Section steps",
              alpha=0.5)

    closest_y = f(x_min)

    plt.scatter(x_min, closest_y,
                color="purple", s=100, zorder=6,
                label=f"Closest point ({x_min:.4f}, {closest_y:.4f})")

    plt.plot([px, x_min], [py, closest_y],
              "k--", linewidth=2,
              label="Shortest distance")

    plt.annotate(
        f"P = ({px}, {py})",
        (px, py),
        xytext=(10, 10),
        textcoords="offset points"
    )

    plt.annotate(
        f"Closest point\n({x_min:.4f}, {closest_y:.4f})",
        (x_min, closest_y),
        xytext=(10, -30),
        textcoords="offset points"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    safe_name = (
        title.replace(' ', '_')
        .replace(',', '')
        .replace('(', '')
        .replace(')', '')
        .replace('/', '-')
        .replace('^', '')
    )
    plt.savefig(f"plot_{safe_name}.png", dpi=120)
    plt.show()


# GENERAL DRIVER
# calculate the shortest distance from a point to a function
# using both Newton-Raphson and Golden Section Search

def find_distance(px, py, f,
                   newton_x0,
                   golden_interval,
                   title,
                   plot_range=(-10, 10),
                   df=None,
                   ddf=None):

    newton_x, newton_dist, newton_history = newton_raphson(
        px, py, f, newton_x0, df=df, ddf=ddf
    )

    golden_x, golden_dist, golden_history, golden_iters = golden_section_search(
        px, py, f,
        golden_interval[0],
        golden_interval[1]
    )

    print("=" * 60)
    print(title)
    print(f"Point: ({px}, {py})")

    print("\nNewton-Raphson:")
    print(f"  x-coordinate = {newton_x:.10f}")
    print(f"  distance     = {newton_dist:.10f}")
    print(f"  iterations   = {len(newton_history) - 1}")

    print("\nGolden Section Search:")
    print(f"  x-coordinate = {golden_x:.10f}")
    print(f"  distance     = {golden_dist:.10f}")
    print(f"  iterations   = {golden_iters}")

    # sanity check: both methods should agree closely
    print(f"\nAgreement check: |newton_x - golden_x| = {abs(newton_x - golden_x):.2e}")
    print(f"                 |newton_dist - golden_dist| = {abs(newton_dist - golden_dist):.2e}")

    plot_search(
        px, py, f,
        newton_history,
        golden_history,
        newton_x,
        title,
        plot_range
    )

    return {
        "newton_x": newton_x,
        "newton_distance": newton_dist,
        "golden_x": golden_x,
        "golden_distance": golden_dist,
    }


# EXAMPLES

if __name__ == "__main__":

    # PART 1: given parabola  y = x^2 + 5
    # analytical derivatives: f'(x) = 2x, f''(x) = 2

    parabola1 = lambda x: x**2 + 5
    parabola1_df = lambda x: 2 * x
    parabola1_ddf = lambda x: 2 * np.ones_like(x) if isinstance(x, np.ndarray) else 2.0

    points = [
        (0, 0),
        (-4, 0),
        (-8, 0),
        (2, 0),
        (6, 0),
    ]

    for px, py in points:
        find_distance(
            px=px, py=py, f=parabola1,
            newton_x0=px,
            golden_interval=(-12, 12),
            title=f"y = x^2 + 5, Point ({px}, {py})",
            plot_range=(-12, 12),
            df=parabola1_df,
            ddf=parabola1_ddf,
        )

    # EXTRA PARABOLA: y = x^2 - 3x + 2
    # analytical derivatives: f'(x) = 2x - 3, f''(x) = 2

    parabola2 = lambda x: x**2 - 3 * x + 2
    parabola2_df = lambda x: 2 * x - 3
    parabola2_ddf = lambda x: 2 * np.ones_like(x) if isinstance(x, np.ndarray) else 2.0

    other_points = [
        (0, 0),
        (5, 4),
        (-3, 2),
    ]

    for px, py in other_points:
        find_distance(
            px=px, py=py, f=parabola2,
            newton_x0=px,
            golden_interval=(-10, 10),
            title=f"y = x^2 - 3x + 2, Point ({px}, {py})",
            plot_range=(-8, 10),
            df=parabola2_df,
            ddf=parabola2_ddf,
        )

    # NON-POLYNOMIAL FUNCTIONS
    # no analytical df/ddf supplied here on purpose -- fall back to
    # numerical derivatives inside newton_raphson, demonstrating the
    # code still works when derivatives aren't hand-derived.

    # exponential: y = e^(0.4x)
    exponential = lambda x: np.exp(0.4 * x)
    find_distance(
        px=2, py=0, f=exponential,
        newton_x0=1,
        golden_interval=(-5, 5),
        title="y = exp(0.4x), Point (2, 0)",
        plot_range=(-5, 5),
    )

    # logarithmic: y = ln(x + 6)   (shifted so domain is valid over test range)
    logarithmic = lambda x: np.log(x + 6)
    find_distance(
        px=-2, py=1, f=logarithmic,
        newton_x0=0,
        golden_interval=(-5.9, 10),
        title="y = ln(x + 6), Point (-2, 1)",
        plot_range=(-5.9, 10),
    )

    # rational (variable in denominator): y = 1 / (x + 3)
    # keep test point/interval away from the x = -3 singularity
    rational = lambda x: 1 / (x + 3)
    find_distance(
        px=1, py=1, f=rational,
        newton_x0=0,
        golden_interval=(-2.5, 6),
        title="y = 1 / (x + 3), Point (1, 1)",
        plot_range=(-2.5, 6),
    )

    # root function: y = sqrt(x + 4)
    # domain requires x >= -4; keep interval/starting guess inside it
    radical = lambda x: np.sqrt(np.maximum(x + 4, 0))
    find_distance(
        px=2, py=3, f=radical,
        newton_x0=0,
        golden_interval=(-3.9, 10),
        title="y = sqrt(x + 4), Point (2, 3)",
        plot_range=(-3.9, 10),
    )