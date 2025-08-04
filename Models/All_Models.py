import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import (mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error)
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from scipy import stats

path = "Master Dataset.xlsx"
df = pd.read_excel(path)
df_pre = df.copy()
df_init = df.shape[0]

def mad_calc():
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        med = df[col].median()
        mad = np.median(np.abs(df[col] - med))
        lower, upper = med - 3*mad, med + 3*mad
        df[col] = df[col].clip(lower=lower, upper=upper)

    arr_pre = df_pre[num_cols].to_numpy()
    arr_post = df[num_cols].to_numpy()
    modified_mask = (arr_pre != arr_post).any(axis=1)
    n_modified = modified_mask.sum()

    print(f"✅ MAD-capping modified {n_modified} rows "
          f"({n_modified/len(df):.1%} of the data)")


# 2. Define target and inputs
target_col = "DieselTotal"
exclude_cols = ["MonthYear", "Truck", "True BCMPerEH"]
numeric_df = df.select_dtypes(include=[np.number])

z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
df = df[(z_scores < 2.5).all(axis=1)]
df_final = df.shape[0]

if target_col not in df.columns:
    raise KeyError(f"Column '{target_col}' not found in the data.")

# 3. Split into features (X) and target (y)
X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

print(f"Filtering removed {df_init - df_final} rows or {round(((df_init-df_final)/df_final*100))}% of data")


print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
print(f"Model inputs (X) has shape {X.shape} and columns:\n   {list(X.columns)}")
print(f"Model target (y) has length {len(y)} and name '{y.name}'")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def mlr_model (X_train, X_test, y_train, y_test, cv):
    mlr = LinearRegression()

    neg_mse = cross_val_score(
        mlr, X_train, y_train, cv = cv,
        scoring="neg_mean_squared_error"
    )

    cv_rmse = np.sqrt(-neg_mse)
    print(f"MLR train-CV RMSE (per fold): {cv_rmse.round(2)}")
    print(f"MLR mean train-CV RMSE: {cv_rmse.mean():.2f}\n")

    mlr.fit(X_train, y_train)

    y_pred = mlr.predict(X_test)
    test_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"MLR test RMSE: {test_rmse:.2f}")
    print(f"MLR test MAE : {test_mae:.2f}")
    print(f"MLR test R²  : {test_r2:.3f}")
    print(f"MLR test MAPE: {test_mape:.1f}%\n")

    return mlr, cv_rmse, test_rmse


def train_and_evaluate(model, X_train, y_train, X_test, y_test, cv=5):
    neg_mse = cross_val_score(
        model, X_train, y_train,
        cv=cv,
        scoring="neg_mean_squared_error"
    )
    cv_rmse = np.sqrt(-neg_mse)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\n=== {model.__class__.__name__} ===")
    print(f"Train-CV RMSE: {cv_rmse.mean():.2f} (±{cv_rmse.std():.2f})")
    print(f"Test   RMSE : {test_rmse:.2f}")
    print(f"Test   MAE  : {test_mae:.2f}")
    print(f"Test   R²   : {test_r2:.3f}")
    print(f"Test   MAPE : {test_mape:.1f}%")

model_mlr, mlr_cv_scores, mlt_test_rmse = mlr_model(X_train, X_test, y_train, y_test, 5) #-- MLR Model as BaseLine

models = {
    "KNN" : KNeighborsRegressor(
    n_neighbors=18,       # number of nearest neighbors to use
    weights="distance",   # or "distance" (closer neighbors weighted more)
    algorithm="auto",    # "ball_tree", "kd_tree", "brute"
    leaf_size=30,        # size of leaf in tree-based algorithms
    p=1,                 # power parameter: 1 = Manhattan, 2 = Euclidean
    metric="minkowski",  # distance function (e.g. "euclidean", "manhattan")
    n_jobs=-1            # use all CPUs
    ),

    "RF" : RandomForestRegressor(
    n_estimators=100,          # number of trees
    max_depth=None,            # limit tree depth (try 5–30)
    min_samples_split=2,       # minimum samples required to split an internal node
    min_samples_leaf=1,        # min samples required at a leaf
    max_features="sqrt",       # number of features considered at each split ("sqrt", "log2", float)
    bootstrap=True,            # use bootstrap samples
    oob_score=False,           # compute out-of-bag score (if bootstrap=True)
    random_state=42,
    n_jobs=-1                  # use all CPUs
    ),

    "XGB" : xgb.XGBRegressor(
    learning_rate=0.1,         # step size shrinkage (try 0.01–0.3)
    n_estimators=100,          # number of boosting rounds (trees)
    max_depth=6,               # max tree depth (try 3–10)
    min_child_weight=1,        # min sum of instance weight needed in a child
    subsample=1.0,             # fraction of rows to sample for each tree
    colsample_bytree=1.0,      # fraction of features to use for each tree
    gamma=0,                   # min loss reduction to make a split (try 0–5)
    reg_alpha=0,               # L1 regularization (try 0–1)
    reg_lambda=1,              # L2 regularization (try 0–1)
    objective="reg:squarederror",
    booster="gbtree",
    tree_method="hist",
    random_state=42,
    n_jobs=-1
    )
}
for name, mdl in models.items():
    train_and_evaluate(mdl, X_train, y_train, X_test, y_test, cv=5)


