"""ElasticNet modeling workflow for MineMetrics.

This script follows the agreed plan for building an interpretable ElasticNet
baseline:

1. Load the existing "Master Dataset.xlsx" fuel consumption dataset and align
   the target with the other models (``DieselTotal``).
2. Inspect columns, report missing values, and clip extreme target outliers to
   limit their influence on the squared-error objective.
3. Assemble a preprocessing pipeline that imputes missing values, one-hot
   encodes categorical predictors, scales numeric variables, and drops
   zero-variance features before fitting the regression model.
4. Wrap the preprocessing transformer and ElasticNet estimator in a single
   pipeline so the entire workflow can be tuned with ``GridSearchCV``.
5. Evaluate the tuned model on a held-out test set and export diagnostics such
   as evaluation metrics, coefficient tables, and prediction plots.

The produced artefacts serve as an explainable baseline that complements the
non-linear models already present in the repository.
"""

from __future__ import annotations

from pathlib import Path
import json

try:  # Optional dependency for plotting
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - graceful fallback if unavailable
    plt = None
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.feature_selection import VarianceThreshold


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path("Master Dataset.xlsx")
TARGET_COLUMN = "DieselTotal"
EXCLUDE_COLUMNS = ["MonthYear", "Truck", "True BCMPerEH"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Cap extreme targets to the 1st/99th percentile by default. Adjust the tuple
# to tune aggressiveness or disable by setting to ``None``.
TARGET_CLIP_QUANTILES: tuple[float, float] | None = (0.01, 0.99)

# Optional polynomial features on numeric predictors. Disabled by default but
# keeping the hook in case mild non-linear effects need to be modeled.
INCLUDE_POLYNOMIALS = False
POLYNOMIAL_DEGREE = 2
POLYNOMIAL_INTERACTION_ONLY = False

# Hyperparameter grid for ElasticNet tuning.
ALPHA_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
L1_RATIO_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]

# Output directory for metrics, plots, and coefficient tables.
OUTPUT_DIR = Path("Models") / "elasticnet_outputs"


def summarise_dataset(df: pd.DataFrame) -> None:
    """Print a compact dataset overview for traceability."""

    print("\n=== Dataset Overview ===")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if missing_summary.empty:
        print("No missing values detected.")
    else:
        print("Missing values (non-zero columns):")
        print(missing_summary.sort_values(ascending=False))


def clip_target_outliers(series: pd.Series) -> pd.Series:
    """Clip extreme target values using pre-defined quantiles."""

    if TARGET_CLIP_QUANTILES is None:
        return series

    lower_q, upper_q = TARGET_CLIP_QUANTILES
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    clipped = series.clip(lower=lower, upper=upper)
    n_clipped = int((clipped != series).sum())

    print(
        "Target clipping:",
        f"lower quantile ({lower_q:.0%}) = {lower:.2f}",
        f"upper quantile ({upper_q:.0%}) = {upper:.2f}",
    )
    print(f"Clipped {n_clipped} target values (" f"{n_clipped/len(series):.1%} of rows).")

    return clipped


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """Construct the preprocessing transformer for the model pipeline."""

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if INCLUDE_POLYNOMIALS:
        numeric_steps.append(
            (
                "polynomial",
                PolynomialFeatures(
                    degree=POLYNOMIAL_DEGREE,
                    include_bias=False,
                    interaction_only=POLYNOMIAL_INTERACTION_ONLY,
                ),
            )
        )

    numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Create the full modeling pipeline with ElasticNet."""

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("variance_filter", VarianceThreshold(threshold=0.0)),
            (
                "model",
                ElasticNet(
                    max_iter=10_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def run_grid_search(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    """Tune ElasticNet hyperparameters with cross-validation."""

    param_grid = {
        "model__alpha": ALPHA_GRID,
        "model__l1_ratio": L1_RATIO_GRID,
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    print("\n=== Grid Search ===")
    print(f"Best RMSE (CV): {-grid_search.best_score_:.3f}")
    print(f"Best params   : {grid_search.best_params_}")

    return grid_search


def evaluate(best_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate the tuned pipeline on the held-out test data."""

    y_pred = best_pipeline.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Test Set Evaluation ===")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }


def extract_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Extract ElasticNet coefficients aligned with feature names."""

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    variance_filter: VarianceThreshold = pipeline.named_steps["variance_filter"]
    support_mask = variance_filter.get_support()
    filtered_names = np.asarray(feature_names)[support_mask]

    coefs = pipeline.named_steps["model"].coef_

    coef_df = pd.DataFrame(
        {
            "feature": filtered_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return coef_df.reset_index(drop=True)


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    """Create an Actual vs Predicted scatter plot (if matplotlib is available)."""

    if plt is None:
        print("matplotlib is not installed; skipping Actual vs Predicted plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
    plt.xlabel("Actual DieselTotal")
    plt.ylabel("Predicted DieselTotal")
    plt.title("ElasticNet: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_coefficients(coef_df: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    """Plot the largest coefficients for quick visual inspection."""

    if plt is None:
        print("matplotlib is not installed; skipping coefficient plot.")
        return

    top_features = coef_df.head(top_n).iloc[::-1]  # reverse for horizontal bar
    plt.figure(figsize=(10, max(6, top_n * 0.35)))
    plt.barh(top_features["feature"], top_features["coefficient"], color="#1f77b4")
    plt.xlabel("Coefficient value")
    plt.title(f"Top {top_n} ElasticNet Coefficients")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected dataset at {DATA_PATH.resolve()} but file was not found."
        )

    df = pd.read_excel(DATA_PATH)

    summarise_dataset(df)

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Column '{TARGET_COLUMN}' not found in the dataset.")

    feature_df = df.drop(columns=[c for c in EXCLUDE_COLUMNS if c in df.columns])

    X = feature_df.drop(columns=[TARGET_COLUMN])
    y = feature_df[TARGET_COLUMN]

    y = clip_target_outliers(y)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    pipeline = build_pipeline(preprocessor)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    grid_search = run_grid_search(pipeline, X_train, y_train)
    best_pipeline: Pipeline = grid_search.best_estimator_

    metrics = evaluate(best_pipeline, X_test, y_test)

    # Persist evaluation artefacts
    metrics_path = OUTPUT_DIR / "elasticnet_metrics.json"
    metrics_payload = {
        "metrics": metrics,
        "best_params": grid_search.best_params_,
        "best_cv_rmse": float(-grid_search.best_score_),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Saved metrics to {metrics_path}")

    cv_results_path = OUTPUT_DIR / "elasticnet_cv_results.csv"
    pd.DataFrame(grid_search.cv_results_).to_csv(cv_results_path, index=False)
    print(f"Saved full CV results to {cv_results_path}")

    coef_df = extract_coefficients(best_pipeline)
    coef_path = OUTPUT_DIR / "elasticnet_coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Saved coefficient table to {coef_path}")

    y_pred_test = best_pipeline.predict(X_test)
    scatter_path = OUTPUT_DIR / "elasticnet_actual_vs_predicted.png"
    plot_actual_vs_predicted(y_test, y_pred_test, scatter_path)
    print(f"Saved Actual vs Predicted plot to {scatter_path}")

    coef_plot_path = OUTPUT_DIR / "elasticnet_top_coefficients.png"
    plot_coefficients(coef_df, coef_plot_path)
    print(f"Saved coefficient plot to {coef_plot_path}")


if __name__ == "__main__":
    main()

