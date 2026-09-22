"""
Resampling, Model Selection and Regularization with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import pandas as pd
from sklearn.datasets import load_diabetes


def load_data():
    # Setting return_X_y=True allows direct unpacking into (DataFrame, Series)
    X, y = load_diabetes(as_frame=True, return_X_y=True)
    return X, y


def describe_data(X, y):
    n, p = X.shape
    return {
        "n": n,
        "p": p,
        "features": X.columns.tolist(),
        "y_mean": round(float(y.mean()), 2),
    }

# Step 2 - train_test
from sklearn.model_selection import train_test_split

def train_test(X, y, test_size=0.25, random_state=0):
    # TODO: train_test_split; return (X_train, X_test, y_train, y_test)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Step 3 - validation_set_curve
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def poly_model(degree):
    # Pipeline that creates polynomial terms, then fits standard linear regression
    return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())


def validation_set_curve(X, y, feature, degrees, random_state):
    # Select feature as a 2D DataFrame/array slice
    X_feat = X[[feature]]

    # 50/50 train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y, test_size=0.5, random_state=random_state
    )

    curve = {}
    for d in degrees:
        model = poly_model(d)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        curve[d] = round(float(mse), 1)

    return curve


def curve_spread(X, y, feature, degrees, seeds):
    # Collect MSEs across seeds for each degree
    # Structure: {degree: [mse_seed1, mse_seed2, ...]}
    degree_mses = {d: [] for d in degrees}

    for seed in seeds:
        seed_curve = validation_set_curve(
            X, y, feature, degrees, random_state=seed
        )
        for d in degrees:
            degree_mses[d].append(seed_curve[d])

    # Compute spread (max - min) rounded to 1 decimal place
    spread = {
        d: round(float(max(degree_mses[d]) - min(degree_mses[d])), 1)
        for d in degrees
    }

    return spread

# Step 4 - cv_mse
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score


def cv_mse(estimator, X, y, k=5, random_state=0):
    # Set up shuffled KFold splitter
    cv = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # cross_val_score returns negative MSE, so negate it to get positive MSE
    neg_mse_scores = cross_val_score(
        estimator, X, y, cv=cv, scoring="neg_mean_squared_error"
    )
    mse_scores = -neg_mse_scores

    # Mean MSE across folds
    mean_mse = np.mean(mse_scores)

    # Standard error of the mean: SE = std / sqrt(k) with ddof=1 for sample std
    se_mse = np.std(mse_scores, ddof=1) / np.sqrt(k)

    return round(float(mean_mse), 2), round(float(se_mse), 2)


def loocv_mse(estimator, X, y):
    # LeaveOneOut evaluates on each individual point
    loo = LeaveOneOut()

    neg_mse_scores = cross_val_score(
        estimator, X, y, cv=loo, scoring="neg_mean_squared_error"
    )
    mse_scores = -neg_mse_scores

    # Average squared error over all held-out observations
    mean_mse = np.mean(mse_scores)

    return round(float(mean_mse), 2)

# Step 5 - cv_spread_by_k
import numpy as np


def cv_spread_by_k(estimator, X, y, ks, seeds):
    out = {}

    for k in ks:
        # Collect the mean CV MSE across each seed for this specific k
        seed_means = []
        for seed in seeds:
            mean_mse, _ = cv_mse(estimator, X, y, k=k, random_state=seed)
            seed_means.append(mean_mse)

        # Average and sample standard deviation across seed runs
        overall_mean = np.mean(seed_means)
        overall_std = np.std(seed_means, ddof=1)

        out[k] = (round(float(overall_mean), 1), round(float(overall_std), 1))

    return out


def compare_with_loocv(estimator, X, y, ks, seeds):
    return {
        "loocv": loocv_mse(estimator, X, y),
        "kfold": cv_spread_by_k(estimator, X, y, ks, seeds),
    }

# Step 6 - bootstrap_coefficients
import numpy as np
from sklearn.linear_model import LinearRegression


def bootstrap_coefficients(X, y, n_boot=200, random_state=0):
    # Ensure inputs support array indexing
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    n, p = X_arr.shape
    rng = np.random.default_rng(random_state)

    coefs = np.empty((n_boot, p))
    model = LinearRegression()

    for b in range(n_boot):
        # Sample n indices with replacement from [0, n)
        idx = rng.integers(0, n, size=n)
        model.fit(X_arr[idx], y_arr[idx])
        coefs[b, :] = model.coef_

    return coefs


def bootstrap_se(coefs):
    # Standard deviation across bootstrap replicates for each coefficient
    se = np.std(coefs, axis=0, ddof=1)
    return np.round(se, 2)


def ols_standard_errors(X, y):
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    n, p = X_arr.shape

    # Design matrix A includes a column of ones for the intercept
    A = np.column_stack([np.ones(n), X_arr])

    # Fit OLS to compute residuals
    model = LinearRegression().fit(X_arr, y_arr)
    y_pred = model.predict(X_arr)
    residuals = y_arr - y_pred

    # Degrees of freedom: n - (p + 1) for p predictors plus intercept
    sigma2 = np.sum(residuals**2) / (n - p - 1)

    # Variance-covariance matrix of coefficients: sigma^2 * (A^T A)^-1
    cov_matrix = sigma2 * np.linalg.inv(A.T @ A)

    # Standard errors are sqrt of diagonal entries; drop index 0 (intercept)
    se = np.sqrt(np.diag(cov_matrix)[1:])

    return np.round(se, 2)

# Step 7 - stepwise_path
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression


def select_features(X, y, k, direction, cv):
    # Initialize SequentialFeatureSelector with LinearRegression and neg MSE scoring
    sfs = SequentialFeatureSelector(
        estimator=LinearRegression(),
        n_features_to_select=k,
        direction=direction,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    sfs.fit(X, y)

    # Use get_feature_names_out to return the selected feature names as a list
    return list(sfs.get_feature_names_out())


def stepwise_path(X, y, direction, cv):
    p = X.shape[1]
    all_cols = list(X.columns)

    path = {}
    for k in range(1, p):
        path[k] = select_features(X, y, k=k, direction=direction, cv=cv)

    # When k == p, all columns are selected by definition
    path[p] = all_cols

    return path

# Step 8 - score_path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


def score_path(X, y, path, cv):
    # Sort keys to ensure sizes are evaluated in ascending order
    sizes = sorted(path.keys())
    means = []
    ses = []

    model = LinearRegression()

    for k in sizes:
        features = path[k]
        # Cross-validation over the selected subset of features
        scores = cross_val_score(
            model,
            X[features],
            y,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        # Convert negative MSE to standard positive MSE
        mse_scores = -scores

        means.append(np.mean(mse_scores))
        # Standard error: sample std / sqrt(n_folds)
        n_folds = len(mse_scores)
        se = (
            np.std(mse_scores, ddof=1) / np.sqrt(n_folds)
            if n_folds > 1
            else 0.0
        )
        ses.append(se)

    return sizes, means, ses


def best_size(sizes, means):
    # Find the size corresponding to the minimum mean MSE
    best_idx = np.argmin(means)
    return sizes[best_idx]

# Step 9 - one_se_rule
import numpy as np


def one_se_rule(values, means, ses, prefer="smaller"):
    values = np.asarray(values)
    means = np.asarray(means)
    ses = np.asarray(ses)

    # Locate minimum mean and calculate the 1-SE threshold
    best_idx = np.argmin(means)
    threshold = means[best_idx] + ses[best_idx]

    # Filter candidate values whose mean error is within 1 SE of the minimum
    valid_mask = means <= threshold
    candidates = values[valid_mask]

    # Select the simplest (smallest) or most expressive (largest) candidate
    if prefer == "smaller":
        return np.min(candidates)
    elif prefer == "larger":
        return np.max(candidates)
    else:
        raise ValueError("prefer must be either 'smaller' or 'larger'")


def choose_subset(X, y, direction, cv):
    # 1. Generate feature selection path
    path = stepwise_path(X, y, direction=direction, cv=cv)

    # 2. Score each subset size along the path
    sizes, means, ses = score_path(X, y, path, cv=cv)

    # 3. Find minimum MSE size and 1-SE rule size
    size_min = best_size(sizes, means)
    size_1se = one_se_rule(sizes, means, ses, prefer="smaller")

    # 4. Retrieve features for the chosen 1-SE model size
    features_1se = path[size_1se]

    return size_min, size_1se, features_1se

# Step 10 - ridge_path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ridge_model(alpha):
    # Pipeline that standardizes features then applies Ridge regression
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def ridge_path(X, y, alphas):
    p = X.shape[1]
    coef_matrix = np.empty((len(alphas), p))

    for i, a in enumerate(alphas):
        model = ridge_model(a)
        model.fit(X, y)
        # Extract the coefficients from the Ridge step in the pipeline
        coef_matrix[i, :] = model.named_steps["ridge"].coef_

    return coef_matrix


def coef_norms(path):
    # Calculate Euclidean (L2) norm across columns for each row (alpha)
    norms = np.linalg.norm(path, ord=2, axis=1)
    return np.round(norms, 2)

# Step 11 - cv_curve
import numpy as np
from sklearn.model_selection import cross_val_score


def cv_curve(make_model, X, y, values, cv):
    means = []
    ses = []

    for val in values:
        model = make_model(val)
        scores = cross_val_score(
            model, X, y, scoring="neg_mean_squared_error", cv=cv, n_jobs=-1
        )
        mse_scores = -scores

        # Mean MSE across folds
        means.append(np.mean(mse_scores))

        # Standard error: sample std (ddof=1) / sqrt(n_folds)
        n_folds = len(mse_scores)
        se = (
            np.std(mse_scores, ddof=1) / np.sqrt(n_folds)
            if n_folds > 1
            else 0.0
        )
        ses.append(se)

    return np.round(means, 1), np.round(ses, 1)


def choose_penalty(make_model, X, y, values, cv):
    values = np.asarray(values)

    # 1. Compute CV error curve and SEs across the penalty grid
    means, ses = cv_curve(make_model, X, y, values, cv)

    # 2. Identify the penalty corresponding to minimum CV MSE
    best_idx = np.argmin(means)
    val_min = values[best_idx]

    # 3. Apply 1-SE rule preferring a larger penalty (stronger regularization)
    val_1se = one_se_rule(values, means, ses, prefer="larger")

    return val_min, val_1se

# Step 12 - lasso_path
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def lasso_model(alpha):
    # Pipeline that standardizes features then applies Lasso regression
    return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=20000))


def lasso_path(X, y, alphas):
    p = X.shape[1]
    coef_matrix = np.empty((len(alphas), p))

    for i, a in enumerate(alphas):
        model = lasso_model(a)
        model.fit(X, y)
        coef_matrix[i, :] = model.named_steps["lasso"].coef_

    return coef_matrix


def nonzero_features(coef, names, tol=1e-8):
    coef = np.asarray(coef)
    names = np.asarray(names)

    mask = np.abs(coef) > tol
    # .tolist() casts np.str_ scalars to native Python str objects
    return names[mask].tolist()


def lasso_cv(X, y, cv):
    # Standardize within pipeline before running cross-validated Lasso
    pipe = make_pipeline(
        StandardScaler(),
        LassoCV(cv=cv, max_iter=20000, random_state=0, n_jobs=-1),
    )
    pipe.fit(X, y)

    lasso_step = pipe.named_steps["lassocv"]
    best_alpha = round(float(lasso_step.alpha_), 4)

    # Feature names can be inferred from X if it's a DataFrame, otherwise fall back to string indices
    feature_names = (
        X.columns.tolist() if hasattr(X, "columns") else [f"x{i}" for i in range(X.shape[1])]
    )
    selected_features = nonzero_features(lasso_step.coef_, feature_names)

    return best_alpha, selected_features

# Step 13 - pcr_model (not yet solved)
# TODO: implement

# Step 14 - pls_model (not yet solved)
# TODO: implement

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

