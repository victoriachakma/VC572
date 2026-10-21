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

def create_polynomial_design_matrix(X, degree):
    if X.ndim == 1:
        X = X[:, np.newaxis]
    N, D = X.shape
    features = [np.ones((N, 1))]
    for d in range(1, degree + 1):
        for col in range(D):
            features.append(X[:, col:col+1]**d)
    return np.hstack(features)

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

if __name__ == "__main__":
    print("=" * 70)
    print("SYDE 572 - Assignment 2, Part 1: Polynomial Basis Function Regression")
    print("=" * 70)

    X, y = load_boston_data()
    X_train_raw, y_train = X[:100], y[:100]
    X_test_raw, y_test = X[100:], y[100:]
    
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std == 0] = 1.0
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    # --- Part a ---
    print("\n[Part a] Evaluating Polynomial Degrees 1 through 7 (Unregularized):")
    print(f"{'Degree':<8} | {'Training RMSE':<15} | {'Test RMSE':<15}")
    print("-" * 45)
    train_errors, test_errors = [], []
    degrees = range(1, 8)
    for deg in degrees:
        Phi_tr = create_polynomial_design_matrix(X_train, deg)
        Phi_te = create_polynomial_design_matrix(X_test, deg)
        w = train_linear_regression(Phi_tr, y_train, lambd=0.0)
        tr_err = rmse(y_train, Phi_tr @ w)
        te_err = rmse(y_test, Phi_te @ w)
        train_errors.append(tr_err)
        test_errors.append(te_err)
        print(f"{deg:<8} | {tr_err:<15.4f} | {te_err:<15.4e}")

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_errors, "o-", color="blue", linewidth=2, label="Training Error")
    plt.plot(degrees, test_errors, "s-", color="red", linewidth=2, label="Test Error")
    plt.yscale("log")
    plt.xlabel("Polynomial Degree (M)")
    plt.ylabel("Root Mean Squared Error (Log Scale)")
    plt.title("Part 1a: Polynomial Regression - Train vs Test Error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/plot_a2_part1_errors.png", dpi=120)
    plt.close()

    # --- Part b ---
    print("\n[Part b] Inspecting Degree 1 Model Weights (All Features):")
    Phi_tr_d1 = create_polynomial_design_matrix(X_train, 1)
    w_d1 = train_linear_regression(Phi_tr_d1, y_train, lambd=0.0)
    print(f"  w0 (Bias intercept): {w_d1[0]:.4f}")
    for idx in range(1, len(w_d1)):
        print(f"  w{idx} (Feature {idx}): {w_d1[idx]:.4f}")
    print(f"  -> RM (6th feature, index 6): {w_d1[6]:.4f}")
    print(f"  -> DIS (8th feature, index 8): {w_d1[8]:.4f}")

    # --- Part c ---
    print("\n[Part c] Generating 1D Polynomial Visualizations (Degrees 1, 4, 8)...")
    X_tr_1d = X_train[:, [1]]
    X_te_1d = X_test[:, [1]]
    x_smooth = np.linspace(X_tr_1d.min() - 0.5, X_tr_1d.max() + 0.5, 200)[:, None]

    for deg, label in [(1, "Degree_1_Underfitting"), (4, "Degree_4_Moderate"), (8, "Degree_8_Overfitting")]:
        Phi_1d = create_polynomial_design_matrix(X_tr_1d, deg)
        w_1d = train_linear_regression(Phi_1d, y_train, lambd=0.0)
        
        plt.figure(figsize=(7, 5))
        plt.scatter(X_tr_1d, y_train, color="blue", s=50, label="Training Data")
        plt.scatter(X_te_1d, y_test, color="orange", s=30, alpha=0.6, label="Test Data")
        plt.plot(x_smooth, create_polynomial_design_matrix(x_smooth, deg) @ w_1d, color="red", linewidth=2.5, label=f"Degree {deg} Fit")
        plt.xlabel("Normalized Feature 2 (INDUS/Zoned)")
        plt.ylabel("Target Value (y)")
        plt.title(f"Part 1c: 1D Polynomial Fit - {label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"media/plot_part1c_{label}.png", dpi=120)
        plt.close()

    # --- Part d ---
    print("\n[Part d] 10-Fold Cross-Validation for L2 Regularization (Degree 8, Feature 2):")
    print(f"{'Lambda (λ)':<12} | {'CV RMSE':<15}")
    print("-" * 30)
    X_tr_f2_raw = X_train_raw[:, [1]]
    lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
    k_folds = 10
    fold_size = len(y_train) // k_folds
    cv_errors = []

    for lambd in lambdas:
        val_errs = []
        for k in range(k_folds):
            v_idx = slice(k * fold_size, (k + 1) * fold_size)
            t_idx = np.concatenate([np.arange(0, k * fold_size), np.arange((k + 1) * fold_size, len(y_train))])
            
            mu, sigma = np.mean(X_tr_f2_raw[t_idx]), np.std(X_tr_f2_raw[t_idx])
            sigma = sigma if sigma > 0 else 1.0
            
            tr_norm = (X_tr_f2_raw[t_idx] - mu) / sigma
            val_norm = (X_tr_f2_raw[v_idx] - mu) / sigma
            
            Phi_tr_cv = create_polynomial_design_matrix(tr_norm, 8)
            Phi_val_cv = create_polynomial_design_matrix(val_norm, 8)
            
            w_cv = train_linear_regression(Phi_tr_cv, y_train[t_idx], lambd=lambd)
            val_errs.append(rmse(y_train[v_idx], Phi_val_cv @ w_cv))
            
        mean_cv = np.mean(val_errs)
        cv_errors.append(mean_cv)
        print(f"{lambd:<12} | {mean_cv:<15.4f}")

    plt.figure(figsize=(8, 5))
    plt.semilogx(lambdas[1:], cv_errors[1:], "o-", color="purple", linewidth=2, label="Regularized ($\lambda > 0$)")
    plt.axhline(cv_errors[0], color="black", linestyle="--", label=f"Unregularized Baseline ($\lambda=0$): {cv_errors[0]:.4f}")
    plt.xlabel("Regularization Parameter ($\lambda$) [Log Scale]")
    plt.ylabel("Cross-Validation RMSE")
    plt.title("Part 1d: 10-Fold CV Error vs. L2 Regularization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("media/plot_a2_part1_cv.png", dpi=120)
    plt.close()
    print("Part 1 execution completed successfully!\n")