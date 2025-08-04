import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
from deap import base, creator, tools
import random
import matplotlib.pyplot as plt
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# === Load and prep data ===
path = "Master Dataset.xlsx"
df = pd.read_excel(path)
df_pre = df.copy()
df_init = df.shape[0]

target_col = "DieselTotal"
exclude_cols = ["MonthYear", "Truck", "True BCMPerEH"]
numeric_df = df.select_dtypes(include=[np.number])

z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
df = df[(z_scores < 2.5).all(axis=1)]
df_final = df.shape[0]

if target_col not in df.columns:
    raise KeyError(f"Column '{target_col}' not found in the data.")

X = df.drop(columns=exclude_cols + [target_col])
y = df[target_col]

print(f"Filtering removed {df_init - df_final} rows or {round(((df_init-df_final)/df_final*100))}% of data")
print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
print(f"Model inputs (X) has shape {X.shape} and columns:\n   {list(X.columns)}")
print(f"Model target (y) has length {len(y)} and name '{y.name}'")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


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


# baseline KNN
models = {
    "KNN": KNeighborsRegressor(
        n_neighbors=5,
        weights="distance",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        n_jobs=-1
    ),
}

for name, mdl in models.items():
    train_and_evaluate(mdl, X_train, y_train, X_test, y_test, cv=5)


# --- Evolutionary search for best feature subset + KNN hyperparams ---
FEATURES = list(X.columns)
NUM_FEATURES = len(FEATURES)

# safe creator creation to avoid redefinition errors in interactive sessions
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Feature flags
toolbox.register("attr_bool", lambda: random.randint(0, 1))
# Hyperparameters
toolbox.register("n_neighbors", lambda: random.randint(1, 30))
toolbox.register("weights", lambda: random.randint(0, 1))  # 0 -> uniform, 1 -> distance
toolbox.register("p", lambda: random.choice([1, 2]))

def init_individual():
    return creator.Individual(
        [toolbox.attr_bool() for _ in range(NUM_FEATURES)] +
        [toolbox.n_neighbors(), toolbox.weights(), toolbox.p()]
    )

toolbox.register("individual", init_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# cache to avoid re-evaluating identical genomes
_eval_cache = {}

def evaluate(individual):
    key = tuple(individual)
    if key in _eval_cache:
        return _eval_cache[key]

    feature_mask = individual[:NUM_FEATURES]
    if sum(feature_mask) == 0:
        return_value = (1e6,)
        _eval_cache[key] = return_value
        return return_value

    selected_features = [f for i, f in enumerate(FEATURES) if feature_mask[i] == 1]
    X_sel = X_train[selected_features]

    n_neighbors = individual[NUM_FEATURES]
    n_neighbors = max(1, min(30, int(n_neighbors)))

    weights = "uniform" if individual[NUM_FEATURES + 1] == 0 else "distance"
    p = individual[NUM_FEATURES + 2]
    p = 1 if p == 1 else 2

    pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        metric="minkowski",
        n_jobs=-1
    ))

    try:
        scores = cross_val_score(
            pipeline,
            X_sel,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_percentage_error",
            error_score="raise"
        )
        cv_mape = -scores.mean() * 100  # percent
    except Exception:
        return_value = (1e5,)
        _eval_cache[key] = return_value
        return return_value

    feature_frac = sum(feature_mask) / NUM_FEATURES
    lambda_feat = 5.0  # penalty weight per fraction of features
    fitness = cv_mape + lambda_feat * feature_frac

    return_value = (fitness,)
    _eval_cache[key] = return_value
    return return_value

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)

def custom_mutate(individual, indpb=0.05, hp_mutpb=0.2):
    for i in range(NUM_FEATURES):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]
    if random.random() < hp_mutpb:
        delta = random.choice([-2, -1, 1, 2])
        individual[NUM_FEATURES] = int(np.clip(individual[NUM_FEATURES] + delta, 1, 30))
    if random.random() < hp_mutpb:
        individual[NUM_FEATURES + 1] = random.randint(0, 1)
    if random.random() < hp_mutpb:
        individual[NUM_FEATURES + 2] = random.choice([1, 2])
    return (individual,)

toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

# — EA CONFIG —
random.seed(42)
POP_SIZE = 50    # tweak population size here
N_GEN = 100      # tweak number of generations here
population = toolbox.population(n=POP_SIZE)

# Hall of Fame
hall_of_fame = tools.HallOfFame(1)

# Stats
stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("min", np.min)
stats.register("avg", np.mean)
logbook = tools.Logbook()
logbook.header = ["gen", "min", "avg"]

# Live plot setup
start_time = time.time()
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
line_min, = ax.plot([], [], label="Min MAPE")
line_avg, = ax.plot([], [], label="Avg MAPE", linestyle='--')
ax.set_xlim(0, N_GEN)
ax.set_ylim(0, 100)
ax.set_xlabel("Generation")
ax.set_ylabel("MAPE (%)")
ax.set_title("Live KNN Evolution")
ax.grid(True)
ax.legend()

# Early stopping
best_so_far = float("inf")
no_improve = 0
patience = 10

for gen in range(N_GEN):
    # Evaluate invalid individuals
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    for ind in invalid_ind:
        ind.fitness.values = toolbox.evaluate(ind)

    # Update hall of fame
    hall_of_fame.update(population)

    # Logging
    fits = [ind.fitness.values[0] for ind in population]
    current_min = min(fits)
    logbook.record(gen=gen, min=current_min, avg=np.mean(fits))

    # Early stopping
    if current_min < best_so_far - 1e-6:
        best_so_far = current_min
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at generation {gen} (no improvement for {patience} gens)")
            break

    # Plot update
    gens_so_far = [l["gen"] for l in logbook]
    min_vals_plot = [l["min"] for l in logbook]
    avg_vals_plot = [l["avg"] for l in logbook]
    line_min.set_data(gens_so_far, min_vals_plot)
    line_avg.set_data(gens_so_far, avg_vals_plot)
    ax.set_xlim(0, N_GEN)
    ax.set_ylim(0, max(30, max(fits) * 1.1))
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

    # Elitism + breeding
    elite_size = 2
    elite = tools.selBest(population, elite_size)
    offspring = toolbox.select(population, len(population) - elite_size)
    offspring = list(map(toolbox.clone, offspring))

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.6:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Mutation
    for mutant in offspring:
        if random.random() < 0.3:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    population[:] = elite + offspring

plt.ioff()
plt.show()
end_time = time.time()

# Final evaluation & hall of fame update
invalid_ind = [ind for ind in population if not ind.fitness.valid]
for ind in invalid_ind:
    ind.fitness.values = toolbox.evaluate(ind)
hall_of_fame.update(population)

best_ind = hall_of_fame[0]

print("\n✅ BEST INDIVIDUAL (evolution):")
print(f"Fitness (MAPE + penalty): {best_ind.fitness.values[0]:.2f}")
print("Selected features:")
print([f for i, f in enumerate(FEATURES) if best_ind[i] == 1])
print("n_neighbors:", int(best_ind[NUM_FEATURES]))
print("weights    :", "uniform" if best_ind[NUM_FEATURES + 1] == 0 else "distance")
print("p          :", 1 if best_ind[NUM_FEATURES + 2] == 1 else 2)
print(f"Runtime    : {end_time - start_time:.1f} seconds")

# === Evaluate best evolved individual with same summary metrics ===
best_feature_mask = best_ind[:NUM_FEATURES]
selected_features = [f for i, f in enumerate(FEATURES) if best_feature_mask[i] == 1]
if len(selected_features) == 0:
    raise RuntimeError("Best individual selected zero features, cannot evaluate KNN.")

X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

best_n_neighbors = int(best_ind[NUM_FEATURES])
best_weights = "uniform" if best_ind[NUM_FEATURES + 1] == 0 else "distance"
best_p = 1 if best_ind[NUM_FEATURES + 2] == 1 else 2

best_model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(
        n_neighbors=best_n_neighbors,
        weights=best_weights,
        p=best_p,
        metric="minkowski",
        n_jobs=-1
    )
)

# Cross-validated RMSE on training data
neg_mse = cross_val_score(
    best_model,
    X_train_sel,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error"
)
cv_rmse = np.sqrt(-neg_mse)

best_model.fit(X_train_sel, y_train)
y_pred = best_model.predict(X_test_sel)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"\n=== EVOLVED KNN ===")
print(f"Selected features ({len(selected_features)}): {selected_features}")
print(f"n_neighbors: {best_n_neighbors}")
print(f"weights    : {best_weights}")
print(f"p          : {best_p}")
print(f"Train-CV RMSE: {cv_rmse.mean():.2f} (±{cv_rmse.std():.2f})")
print(f"Test   RMSE : {test_rmse:.2f}")
print(f"Test   MAE  : {test_mae:.2f}")
print(f"Test   R²   : {test_r2:.3f}")
print(f"Test   MAPE : {test_mape:.1f}%")
