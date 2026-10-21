import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('media', exist_ok=True)

FEATURE_NAMES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
                 "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]


# helper functions

def load_boston_data():
    # use the course-provided housing.data (next to this script) if it's there
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "housing.data")
    if os.path.exists(local):
        data = np.loadtxt(local)
        return data[:, :-1], data[:, -1]   # 13 features, MEDV target
    try:
        data_url = "https://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
        # each town spans two lines: 11 features, then B, LSTAT and MEDV (the target)
        X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        y = raw_df.values[1::2, 2]
        return X, y
    except Exception:
        from sklearn.datasets import fetch_openml
        boston = fetch_openml(name='boston', version=1, as_frame=False)
        return boston.data.astype(float), boston.target.astype(float)


def normalize(X_train_raw, X_other_raw):
    """Standardize both sets using the training set's mean and std."""
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std = np.where(std == 0, 1.0, std)   # CHAS is constant in the first 100 rows
    return (X_train_raw - mean) / std, (X_other_raw - mean) / std


def create_polynomial_design_matrix(X, degree):
    """[1, x1..xD, x1^2..xD^2, ..., x1^M..xD^M] -- monomials only, no cross-terms."""
    if X.ndim == 1:
        X = X[:, np.newaxis]
    N, D = X.shape
    features = [np.ones((N, 1))]
    for d in range(1, degree + 1):
        for col in range(D):
            features.append(X[:, col:col+1] ** d)
    return np.hstack(features)


def train_linear_regression(Phi, y, lambd=0.0):
    if lambd == 0.0:
        w, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        return w
    else:
        I = np.eye(Phi.shape[1])
        I[0, 0] = 0  # do not penalize the bias
        return np.linalg.inv(Phi.T @ Phi + lambd * I) @ Phi.T @ y


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# part a + b: polynomial_regression

def polynomial_regression(X_train, y_train, X_test, y_test, degrees=range(1, 8)):
    """Fit unregularized polynomial basis regression for each degree and plot
    train/test RMS error versus degree."""
    print("\n[Part a] Polynomial degrees 1 to 7 (unregularized):")
    print(f"{'Degree':<8} | {'Training RMSE':<15} | {'Test RMSE':<15}")
    print("-" * 45)
    train_errors, test_errors = [], []
    for deg in degrees:
        Phi_tr = create_polynomial_design_matrix(X_train, deg)
        Phi_te = create_polynomial_design_matrix(X_test, deg)
        w = train_linear_regression(Phi_tr, y_train)
        tr_err = rmse(y_train, Phi_tr @ w)
        te_err = rmse(y_test, Phi_te @ w)
        train_errors.append(tr_err)
        test_errors.append(te_err)
        print(f"{deg:<8} | {tr_err:<15.4f} | {te_err:<15.4e}")

    plt.figure(figsize=(8, 5))
    plt.plot(list(degrees), train_errors, "o-", color="blue", linewidth=2, label="Training Error")
    plt.plot(list(degrees), test_errors, "s-", color="red", linewidth=2, label="Test Error")
    plt.yscale("log")
    plt.xlabel("Polynomial Degree (M)")
    plt.ylabel("RMS Error (Log Scale)")
    plt.title("Part 1a: Polynomial Regression - Train vs Test Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/plot_a2_part1_errors.png", dpi=120)
    plt.close()
    return train_errors, test_errors


def inspect_degree1_weights(X_train, y_train):
    """Part (b): weights of the degree 1 model (bias is w0)."""
    print("\n[Part b] Degree 1 weights (w0 = bias, wi = feature i):")
    w = train_linear_regression(create_polynomial_design_matrix(X_train, 1), y_train)
    print(f"  w0  (bias): {w[0]:.4f}")
    for i, name in enumerate(FEATURE_NAMES, start=1):
        print(f"  w{i:<3}({name}): {w[i]:.4f}")
    print(f"  -> 6th feature RM : {w[6]:.4f}")
    print(f"  -> 8th feature DIS: {w[8]:.4f}")
    return w


# part c: polynomial_regression_1d_vis

def polynomial_regression_1d_vis(X_train, y_train, X_test, y_test, degrees=(1, 4, 8)):
    """Polynomial regression on feature 2 only (Xn(:,2) -> column index 1),
    plotting training points, test points and the learned curve."""
    print("\n[Part c] 1D polynomial fits on feature 2 (ZN):")
    x_tr = X_train[:, [1]]
    x_te = X_test[:, [1]]
    # span the full range of train + test points so extrapolation is visible
    x_all = np.vstack([x_tr, x_te])
    x_smooth = np.linspace(x_all.min(), x_all.max(), 300)[:, None]
    tags = {1: "Underfitting", 4: "Moderate", 8: "Overfitting"}

    for deg in degrees:
        Phi_tr = create_polynomial_design_matrix(x_tr, deg)
        w = train_linear_regression(Phi_tr, y_train)
        tr_err = rmse(y_train, Phi_tr @ w)
        te_err = rmse(y_test, create_polynomial_design_matrix(x_te, deg) @ w)
        print(f"  Degree {deg}: train RMSE {tr_err:.3f}, test RMSE {te_err:.3f}")

        plt.figure(figsize=(7, 5))
        plt.scatter(x_tr, y_train, color="blue", s=50, label="Training Data")
        plt.scatter(x_te, y_test, color="orange", s=30, alpha=0.6, label="Test Data")
        plt.plot(x_smooth, create_polynomial_design_matrix(x_smooth, deg) @ w,
                 color="red", linewidth=2.5, label=f"Degree {deg} Fit")
        plt.xlabel("Normalized Feature 2 (ZN: % residential land zoned for large lots)")
        plt.ylabel("Median Value (MEDV, $1000s)")
        plt.ylim(0, 60)  # keep the data readable; high-degree curves shoot off-scale at the edges
        plt.title(f"Part 1c: 1D Polynomial Fit on ZN - Degree {deg}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"media/plot_part1c_Degree_{deg}_{tags.get(deg, 'Fit')}.png", dpi=120)
        plt.close()


# part d: polynomial_regression_reg

def polynomial_regression_reg(X_train_raw, y_train, degree=8,
                              lambdas=(0, 0.01, 0.1, 1, 10, 100, 1000), k_folds=10):
    """L2-regularized degree 8 polynomial on feature 2, lambda chosen by
    10-fold cross-validation. Normalization is recomputed inside each fold."""
    print("\n[Part d] 10-fold CV for L2 regularization (degree 8, feature 2):")
    print(f"{'Lambda':<12} | {'CV RMSE':<15}")
    print("-" * 30)
    x = X_train_raw[:, [1]]
    N = len(y_train)
    fold_size = N // k_folds
    cv_errors = []

    for lambd in lambdas:
        val_errs = []
        for k in range(k_folds):
            v_idx = np.arange(k * fold_size, (k + 1) * fold_size)
            t_idx = np.setdiff1d(np.arange(N), v_idx)
            x_tr, x_val = normalize(x[t_idx], x[v_idx])
            w = train_linear_regression(create_polynomial_design_matrix(x_tr, degree),
                                        y_train[t_idx], lambd=lambd)
            pred = create_polynomial_design_matrix(x_val, degree) @ w
            val_errs.append(rmse(y_train[v_idx], pred))
        cv_errors.append(np.mean(val_errs))
        print(f"{lambd:<12} | {cv_errors[-1]:<15.4f}")

    best = lambdas[int(np.argmin(cv_errors))]
    print(f"  -> Best lambda: {best}")

    plt.figure(figsize=(8, 5))
    plt.semilogx(lambdas[1:], cv_errors[1:], "o-", color="purple", linewidth=2,
                 label=r"Regularized ($\lambda > 0$)")
    plt.axhline(cv_errors[0], color="black", linestyle="--",
                label=rf"Unregularized baseline ($\lambda=0$): {cv_errors[0]:.1f}")
    plt.yscale("log")  # lambda=0 baseline is orders of magnitude above the rest
    plt.xlabel(r"Regularization Parameter ($\lambda$) [Log Scale]")
    plt.ylabel("Average Validation RMSE (Log Scale)")
    plt.title("Part 1d: 10-Fold CV Error vs. L2 Regularization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/plot_a2_part1_cv.png", dpi=120)
    plt.close()
    return cv_errors, best


if __name__ == "__main__":
    print("=" * 70)
    print("SYDE 572 - Assignment 2, Part 1: Polynomial Basis Function Regression")
    print("=" * 70)

    X, y = load_boston_data()
    X_train_raw, y_train = X[:100], y[:100]
    X_test_raw, y_test = X[100:], y[100:]
    X_train, X_test = normalize(X_train_raw, X_test_raw)

    polynomial_regression(X_train, y_train, X_test, y_test)
    inspect_degree1_weights(X_train, y_train)
    polynomial_regression_1d_vis(X_train, y_train, X_test, y_test)
    polynomial_regression_reg(X_train_raw, y_train)
    print("\nPart 1 complete.")
