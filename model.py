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

# Step 5 - cv_spread_by_k (not yet solved)
# TODO: implement

# Step 6 - bootstrap_coefficients (not yet solved)
# TODO: implement

# Step 7 - stepwise_path (not yet solved)
# TODO: implement

# Step 8 - score_path (not yet solved)
# TODO: implement

# Step 9 - one_se_rule (not yet solved)
# TODO: implement

# Step 10 - ridge_path (not yet solved)
# TODO: implement

# Step 11 - cv_curve (not yet solved)
# TODO: implement

# Step 12 - lasso_path (not yet solved)
# TODO: implement

# Step 13 - pcr_model (not yet solved)
# TODO: implement

# Step 14 - pls_model (not yet solved)
# TODO: implement

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

