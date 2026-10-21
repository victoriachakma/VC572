import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('media', exist_ok=True)

def load_boston_data():
    try:
        data_url = "https://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        return data[:, :-1], data[:, -1]
    except Exception:
        from sklearn.datasets import fetch_openml
        boston = fetch_openml(name='boston', version=1, as_frame=False)
        return boston.data, boston.target

def train_linear_regression(Phi, y, lambd=0.0):
    if lambd == 0.0:
        w, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        return w
    else:
        I = np.eye(Phi.shape[1])
        I[0, 0] = 0
        return np.linalg.inv(Phi.T @ Phi + lambd * I) @ Phi.T @ y

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def create_gaussian_design_matrix(X, centers, s=2.0):
    N = X.shape[0]
    K = centers.shape[0]
    Phi = np.ones((N, K + 1))
    for k in range(K):
        dist_sq = np.sum((X - centers[k]) ** 2, axis=1)
        Phi[:, k + 1] = np.exp(-dist_sq / (2 * (s ** 2)))
    return Phi

if __name__ == "__main__":
    print("=" * 70)
    print("SYDE 572 - Assignment 2, Part 2: Gaussian Basis Function Regression")
    print("=" * 70)

    X, y = load_boston_data()
    X_train_raw, y_train = X[:100], y[:100]
    X_test_raw, y_test = X[100:], y[100:]
    
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std == 0] = 1.0
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    # --- Part e ---
    print("\n[Part e] Gaussian Regression: Evaluating Basis Counts K (s=2.0):")
    print(f"{'Basis Count (K)':<15} | {'Training RMSE':<15} | {'Test RMSE':<15}")
    print("-" * 51)
    K_values = range(5, 100, 10)
    train_errors, test_errors = [], []
    np.random.seed(42)
    perm = np.random.permutation(len(X_train))
    X_tr_shuff = X_train[perm]
    s_val = 2.0

    for K in K_values:
        centers = X_tr_shuff[:K]
        Phi_tr = create_gaussian_design_matrix(X_train, centers, s=s_val)
        Phi_te = create_gaussian_design_matrix(X_test, centers, s=s_val)
        w = train_linear_regression(Phi_tr, y_train, lambd=0.0)
        tr_e = rmse(y_train, Phi_tr @ w)
        te_e = rmse(y_test, Phi_te @ w)
        train_errors.append(tr_e)
        test_errors.append(te_e)
        print(f"{K:<15} | {tr_e:<15.4f} | {te_e:<15.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(list(K_values), train_errors, "o-", color="blue", linewidth=2, label="Training RMSE")
    plt.plot(list(K_values), test_errors, "s-", color="red", linewidth=2, label="Test RMSE")
    plt.xlabel("Number of Basis Functions (K)")
    plt.ylabel("Root Mean Squared Error")
    plt.title("Part 2e: Gaussian Basis Regression - Error vs. K (s=2.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/plot_a2_part2_gaussian_errors.png", dpi=120)
    plt.close()

    # --- Part f ---
    print("\n[Part f] Leakage-Free 10-Fold CV for Gaussian L2 Regularization (K=90, s=2.0):")
    print(f"{'Lambda (λ)':<12} | {'CV RMSE':<15}")
    print("-" * 30)
    K = 90
    lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
    k_folds = 10
    fold_size = len(y_train) // k_folds
    cv_errors = []

    for lambd in lambdas:
        val_errs = []
        for k in range(k_folds):
            v_idx = slice(k * fold_size, (k + 1) * fold_size)
            t_idx = np.concatenate([np.arange(0, k * fold_size), np.arange((k + 1) * fold_size, len(y_train))])
            
            fold_centers = X_train[t_idx][:K]
            Phi_tr_f = create_gaussian_design_matrix(X_train[t_idx], fold_centers, s=s_val)
            Phi_val_f = create_gaussian_design_matrix(X_train[v_idx], fold_centers, s=s_val)
            
            w_cv = train_linear_regression(Phi_tr_f, y_train[t_idx], lambd=lambd)
            val_errs.append(rmse(y_train[v_idx], Phi_val_f @ w_cv))
            
        mean_cv = np.mean(val_errs)
        cv_errors.append(mean_cv)
        print(f"{lambd:<12} | {mean_cv:<15.4f}")

    plt.figure(figsize=(8, 5))
    plt.semilogx(lambdas[1:], cv_errors[1:], "o-", color="darkgreen", linewidth=2, label="Regularized ($\lambda > 0$)")
    plt.axhline(cv_errors[0], color="black", linestyle="--", label=f"Unregularized Baseline ($\lambda=0$): {cv_errors[0]:.4f}")
    plt.xlabel("Regularization Parameter ($\lambda$) [Log Scale]")
    plt.ylabel("Cross-Validation RMSE")
    plt.title("Part 2f: 10-Fold CV Error vs. L2 Regularization (Gaussian Bases, K=90)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/plot_a2_part2_gaussian_cv.png", dpi=120)
    plt.close()
    print("Part 2 execution completed successfully!\n")