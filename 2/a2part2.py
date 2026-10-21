import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('media', exist_ok=True)

# helper functions

def load_boston_data():
    try:
        data_url = "https://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        return data[:, :-1], data[:, -1]
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


def create_gaussian_design_matrix(X, centers, s=2.0):
    """Bias column + one Gaussian bump exp(-||x - mu_j||^2 / 2s^2) per centre."""
    N, K = X.shape[0], centers.shape[0]
    Phi = np.ones((N, K + 1))
    for k in range(K):
        dist_sq = np.sum((X - centers[k]) ** 2, axis=1)
        Phi[:, k + 1] = np.exp(-dist_sq / (2 * s ** 2))
    return Phi


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


# part e: gaussian_regression

def gaussian_regression(X_train, y_train, X_test, y_test,
                        K_values=range(5, 100, 10), s=2.0, seed=42):
    """Unregularized Gaussian basis regression for K = 5, 15, ..., 95.
    Centres = first K points of one random permutation of the training set."""
    print(f"\n[Part e] Gaussian regression, K = 5..95 (s = {s}):")
    print(f"{'K':<6} | {'Training RMSE':<15} | {'Test RMSE':<15}")
    print("-" * 42)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_train))
    train_errors, test_errors = [], []

    for K in K_values:
        centers = X_train[perm[:K]]
        Phi_tr = create_gaussian_design_matrix(X_train, centers, s)
        Phi_te = create_gaussian_design_matrix(X_test, centers, s)
        w = train_linear_regression(Phi_tr, y_train)
        train_errors.append(rmse(y_train, Phi_tr @ w))
        test_errors.append(rmse(y_test, Phi_te @ w))
        print(f"{K:<6} | {train_errors[-1]:<15.4f} | {test_errors[-1]:<15.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_values), train_errors, "o-", color="blue", linewidth=2, label="Training RMSE")
    plt.plot(list(K_values), test_errors, "s-", color="red", linewidth=2, label="Test RMSE")
    plt.xlabel("Number of Basis Functions (K)")
    plt.ylabel("RMS Error")
    plt.title(f"Part 2e: Gaussian Basis Regression - Error vs. K (s={s})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/plot_a2_part2_gaussian_errors.png", dpi=120)
    plt.close()
    return train_errors, test_errors


# part f: gaussian_regression_reg

def gaussian_regression_reg(X_train, y_train, K=90, s=2.0,
                            lambdas=(0, 0.01, 0.1, 1, 10, 100, 1000), k_folds=10, seed=42):
    """L2-regularized Gaussian regression with K = 90 on all features,
    lambda chosen by 10-fold cross-validation."""
    print(f"\n[Part f] 10-fold CV for L2 regularization (K = {K}, s = {s}):")
    print(f"{'Lambda':<12} | {'CV RMSE':<15}")
    print("-" * 30)
    rng = np.random.RandomState(seed)
    N = len(y_train)
    fold_size = N // k_folds
    cv_errors = []

    for lambd in lambdas:
        val_errs = []
        for k in range(k_folds):
            v_idx = np.arange(k * fold_size, (k + 1) * fold_size)
            t_idx = np.setdiff1d(np.arange(N), v_idx)
            # centres come only from this fold's training points; with 90 training
            # points per fold and K = 90, every training point ends up as a centre
            centers = X_train[t_idx][rng.permutation(len(t_idx))[:K]]
            Phi_tr = create_gaussian_design_matrix(X_train[t_idx], centers, s)
            Phi_val = create_gaussian_design_matrix(X_train[v_idx], centers, s)
            w = train_linear_regression(Phi_tr, y_train[t_idx], lambd=lambd)
            val_errs.append(rmse(y_train[v_idx], Phi_val @ w))
        cv_errors.append(np.mean(val_errs))
        print(f"{lambd:<12} | {cv_errors[-1]:<15.4f}")

    best = lambdas[int(np.argmin(cv_errors))]
    print(f"  -> Best lambda: {best}")

    plt.figure(figsize=(8, 5))
    plt.semilogx(lambdas[1:], cv_errors[1:], "o-", color="darkgreen", linewidth=2,
                 label=r"Regularized ($\lambda > 0$)")
    plt.axhline(cv_errors[0], color="black", linestyle="--",
                label=rf"Unregularized baseline ($\lambda=0$): {cv_errors[0]:.2f}")
    plt.xlabel(r"Regularization Parameter ($\lambda$) [Log Scale]")
    plt.ylabel("Average Validation RMSE")
    plt.title(f"Part 2f: 10-Fold CV Error vs. L2 Regularization (Gaussian Bases, K={K})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/plot_a2_part2_gaussian_cv.png", dpi=120)
    plt.close()
    return cv_errors, best


if __name__ == "__main__":
    print("=" * 70)
    print("SYDE 572 - Assignment 2, Part 2: Gaussian Basis Function Regression")
    print("=" * 70)

    X, y = load_boston_data()
    X_train_raw, y_train = X[:100], y[:100]
    X_test_raw, y_test = X[100:], y[100:]
    X_train, X_test = normalize(X_train_raw, X_test_raw)

    gaussian_regression(X_train, y_train, X_test, y_test)
    gaussian_regression_reg(X_train, y_train)
    print("\nPart 2 complete.")
