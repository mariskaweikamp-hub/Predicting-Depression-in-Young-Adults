# ======================== 1. Data Loading & Initial Inspection ======================== #

import pandas as pd
import numpy as np
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from catboost import CatBoostClassifier, Pool, cv as catboost_cv
import optuna
import shap

print("# ======================== 1. Data Loading ======================== #")

# Load SPSS file
file_path = "21600-0008-Data.sav"
df, meta = pyreadstat.read_sav(file_path)

# Build dictionary mapping variable name → SPSS label safely
if hasattr(meta, "column_names") and hasattr(meta, "column_labels"):
    # column_labels is often a list, so zip is the safest
    spss_labels = dict(zip(meta.column_names, meta.column_labels))
else:
    print("Warning: SPSS labels not found in metadata.")
    spss_labels = {col: "Label not found" for col in df.columns}


# Check target variable
if "H3ID15" not in df.columns:
    raise ValueError("Target variable H3ID15 not found in the dataset.")

print("\nTarget variable distribution (H3ID15):")
print(df["H3ID15"].value_counts(dropna=False))

print("\nAverage % missing per column:")
print(df.isna().mean().mean() * 100)


# ======================== 2. Data Quality Visualisation ======================== #

print("\n# ======================== 2. Data Quality Visualisation ======================== #")

# Frequency table for depression outcome
freq_table = (
    df["H3ID15"]
    .value_counts(dropna=False)
    .rename_axis("DepressionDiagnosis")
    .reset_index(name="Count")
)
freq_table["Percent"] = freq_table["Count"] / freq_table["Count"].sum() * 100
print("\nFrequency table for depression outcome (H3ID15):")
print(freq_table)

# Missingness distribution
missing_percent = df.isna().mean() * 100
missing_nonzero = missing_percent[missing_percent > 0]

plt.figure(figsize=(10, 6))
sns.histplot(missing_nonzero, bins=20, kde=False)
plt.title("Distribution of Missingness Across Variables")
plt.xlabel("Percentage of Missing Values per Variable")
plt.ylabel("Number of Variables")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

print("\nTop 10 variables with highest missingness:")
print(missing_percent.sort_values(ascending=False).head(10))


# ======================== 3. Sample Characteristics ======================== #

print("\n# ======================== 3. Sample Characteristics ======================== #")

# Restrict to non-missing outcome
df_sample = df.dropna(subset=["H3ID15"]).copy()
n_total = len(df_sample)
print(f"Total sample size (non-missing H3ID15): {n_total}")

# Sex
sex_map = {1: "Male", 2: "Female"}
sex_counts = df_sample["BIO_SEX3"].map(sex_map).value_counts(dropna=False)

# Age
age_mean = df_sample["CALCAGE3"].mean()
age_sd = df_sample["CALCAGE3"].std()

# Depression prevalence (assuming H3ID15 is coded 0/1)
dep_prev = df_sample["H3ID15"].mean() * 100

summary_table = pd.DataFrame({
    "Variable": ["Total sample", "Male", "Female", "Mean age (SD)", "Depression prevalence (%)"],
    "Value": [
        n_total,
        f"{sex_counts.get('Male', 0)} ({round(sex_counts.get('Male', 0) / n_total * 100, 1)}%)",
        f"{sex_counts.get('Female', 0)} ({round(sex_counts.get('Female', 0) / n_total * 100, 1)}%)",
        f"{age_mean:.1f} ({age_sd:.1f})",
        f"{dep_prev:.1f}"
    ]
})

print("\nSample characteristics:")
print(summary_table.to_string(index=False))

summary_table.to_csv("sample_characteristics.csv", index=False)


# ======================== 4. Preprocessing & Feature Selection ======================== #

print("\n# ======================== 4. Preprocessing & Feature Selection ======================== #")

# Keep only rows with non-missing target
df_clean = df.dropna(subset=["H3ID15"]).copy()
y = df_clean["H3ID15"].astype(int)

# Drop ID and target
X = df_clean.drop(columns=["H3ID15", "AID"], errors="ignore")

# Remove features with >50% missingness
X = X.loc[:, X.isna().mean() < 0.5]
print(f"Number of predictors after missingness filter: {X.shape[1]}")

# Identify numeric / categorical
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [col for col in X.columns if col not in num_cols]

# Impute
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("Missing")

# Encode categorical with ordinal encoder
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
if cat_cols:
    X[cat_cols] = encoder.fit_transform(X[cat_cols])

# Mutual Information feature selection
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\nTop 15 features ranked by mutual information with H3ID15:")
print(mi_series.head(15))

# Keep top 50 features
top_features = mi_series.head(50).index.tolist()
X_top = X[top_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_top, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# ======================== 5. Baseline Models ======================== #

print("\n# ======================== 5. Baseline Models ======================== #")

def evaluate_model(name, model, X_test, y_test):
    """
    Evaluate a fitted classifier on the test set.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"\n{name} performance on test set:")
    print(f"AUC: {auc:.3f} | F1: {f1:.3f} | Balanced Accuracy: {bal_acc:.3f}")

    return {
        "Model": name,
        "AUC_Test": auc,
        "F1": f1,
        "BalancedAcc": bal_acc
    }

baseline_results = []

# 5a. LASSO (Logistic Regression with L1 penalty)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lasso_baseline = LogisticRegression(
    penalty="l1",
    solver="saga",
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)
lasso_baseline.fit(X_train_s, y_train)
baseline_results.append(evaluate_model("LASSO (baseline)", lasso_baseline, X_test_s, y_test))

# 5b. Random Forest (baseline)
rf_baseline = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_estimators=300
)
rf_baseline.fit(X_train, y_train)
baseline_results.append(evaluate_model("Random Forest (baseline)", rf_baseline, X_test, y_test))

# 5c. CatBoost (baseline)
cat_baseline = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=[1, 8],
    random_seed=42,
    verbose=False
)
cat_baseline.fit(X_train, y_train)
baseline_results.append(evaluate_model("CatBoost (baseline)", cat_baseline, X_test, y_test))

baseline_df = pd.DataFrame(baseline_results)
print("\nBaseline model comparison:")
print(baseline_df.to_string(index=False))


# ======================== 6. Hyperparameter Tuning ======================== #

print("\n# ======================== 6. Hyperparameter Tuning ======================== #")

# ---------- 6a. LASSO Tuning ---------- #

print("\n# ======================== 6a. LASSO Tuning ======================== #")

param_grid_lasso = {
    "C": [0.001, 0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"]
}

lasso_grid = GridSearchCV(
    LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=20000,
        random_state=42
    ),
    param_grid=param_grid_lasso,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=1
)

lasso_grid.fit(X_train_s, y_train)
lasso_tuned = lasso_grid.best_estimator_
print("Best LASSO parameters:", lasso_grid.best_params_)
print("Best LASSO CV AUC:", lasso_grid.best_score_)


# ======================== 6b. Random Forest Tuning (Optuna) ======================== #

print("\n# ======================== 6b. Random Forest Tuning (Optuna) ======================== #")

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import optuna

def rf_objective(trial):
    """
    Optuna optimization objective for Random Forest
    using 5-fold stratified cross-validation and AUC.
    """

    # Define hyperparameter search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 60),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": "balanced",     # keep fixed for imbalanced outcome
        "random_state": 42,
        "n_jobs": -1
    }

    # Create model
    model = RandomForestClassifier(**params)

    # Cross-validation AUC
    cv_auc = cross_val_score(
        model, X_train, y_train,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    ).mean()

    # Report back to Optuna
    return cv_auc


# Create study with pruning
sampler = optuna.samplers.TPESampler(seed=42)
rf_study = optuna.create_study(direction="maximize", sampler=sampler)
rf_study.optimize(rf_objective, n_trials=50, show_progress_bar=True)

print("\nBest Optuna parameters for Random Forest:")
print(rf_study.best_params)
print(f"Best Optuna CV AUC: {rf_study.best_value:.4f}")

# Train final model
rf_optuna = RandomForestClassifier(
    **rf_study.best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_optuna.fit(X_train, y_train)

# Quick evaluation on the test set
y_prob_optuna = rf_optuna.predict_proba(X_test)[:, 1]
print(f"Optuna-tuned Random Forest Test AUC: {roc_auc_score(y_test, y_prob_optuna):.4f}")



# ---------- 6c. CatBoost Tuning with Optuna (using CatBoost CV) ---------- #

print("\n# ======================== 6c. CatBoost Tuning (Optuna + CatBoost CV) ======================== #")

def catboost_objective(trial):
    """
    Optuna objective function for CatBoost using CatBoost's
    built-in cross-validation on the training set.
    """
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": trial.suggest_int("iterations", 300, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
        "random_seed": 42,
        "verbose": False,
        "class_weights": [1, 8]
    }

    train_pool = Pool(X_train, y_train)

    cv_results = catboost_cv(
        params=params,
        pool=train_pool,
        fold_count=5,
        shuffle=True,
        stratified=True,
        partition_random_seed=42,
        verbose=False
    )

    best_auc = cv_results["test-AUC-mean"].max()
    return best_auc

sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(catboost_objective, n_trials=40, show_progress_bar=True)

print("\nBest Optuna parameters for CatBoost:")
print(study.best_params)
print(f"Best Optuna CV AUC: {study.best_value:.4f}")

# Train final tuned CatBoost on the full training set
cat_optuna = CatBoostClassifier(
    **study.best_params,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=[1, 8],
    random_seed=42,
    verbose=False
)

cat_optuna.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=80)

# Quick sanity check on test AUC
cat_optuna_prob = cat_optuna.predict_proba(X_test)[:, 1]
print(f"\nFinal Optuna-tuned CatBoost Test AUC: {roc_auc_score(y_test, cat_optuna_prob):.4f}")


# ======================== 7. Tuned Model Evaluation ======================== #

print("\n# ======================== 7. Tuned Model Evaluation ======================== #")

# Make sure tuned models are fitted on original train splits
lasso_tuned.fit(X_train_s, y_train)

# rf_optuna has already been fitted after the Optuna study
# cat_optuna has already been fitted after the Optuna study

tuned_results = []
tuned_results.append(evaluate_model("LASSO (tuned)", lasso_tuned, X_test_s, y_test))
tuned_results.append(evaluate_model("Random Forest (tuned)", rf_optuna, X_test, y_test))
tuned_results.append(evaluate_model("CatBoost (tuned)", cat_optuna, X_test, y_test))


tuned_df = pd.DataFrame(tuned_results)
print("\nTuned model comparison (test set):")
print(tuned_df.to_string(index=False))


# ======================== 8. Retrain Without H3ID26F & H3SP9 ======================== #

print("\n# ======================== 8. Retrain Without H3ID26F & H3SP9 ======================== #")

remove_vars = ["H3ID26F", "H3SP9"]

X_train_red = X_train.drop(columns=[v for v in remove_vars if v in X_train.columns])
X_test_red = X_test.drop(columns=[v for v in remove_vars if v in X_test.columns])

print(f"Reduced train shape: {X_train_red.shape}, Reduced test shape: {X_test_red.shape}")

# LASSO scaling
scaler_red = StandardScaler()
X_train_red_s = scaler_red.fit_transform(X_train_red)
X_test_red_s = scaler_red.transform(X_test_red)

# Refit tuned models on reduced feature set
lasso_red = lasso_tuned
lasso_red.fit(X_train_red_s, y_train)

rf_red = rf_optuna         # <-- FIXED
rf_red.fit(X_train_red, y_train)

cat_red = CatBoostClassifier(
    **study.best_params,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=[1, 8],
    random_seed=42,
    verbose=False
)
cat_red.fit(X_train_red, y_train, eval_set=(X_test_red, y_test), early_stopping_rounds=80)

# Evaluate reduced models
results_reduced = []
results_reduced.append(evaluate_model("LASSO (no H3ID26F/H3SP9)", lasso_red, X_test_red_s, y_test))
results_reduced.append(evaluate_model("Random Forest (no H3ID26F/H3SP9)", rf_red, X_test_red, y_test))
results_reduced.append(evaluate_model("CatBoost (no H3ID26F/H3SP9)", cat_red, X_test_red, y_test))

results_reduced_df = pd.DataFrame(results_reduced)
print("\nPerformance after removing H3ID26F and H3SP9:")
print(results_reduced_df.to_string(index=False))



# ======================== 9. SHAP Analysis (CatBoost, reduced model) ======================== #
print("# ======================== 9. SHAP Analysis (CatBoost, reduced model) ======================== #")

import shap

# Use reduced feature matrix (same features the model was trained on)
X_test_shap = X_test_red.copy()

# Create explainer
explainer_cat = shap.TreeExplainer(cat_red)

# Compute shap values
# For binary CatBoost: the output already matches n_features exactly
shap_values_cat = explainer_cat.shap_values(X_test_shap)

print("SHAP matrix shape:", shap_values_cat.shape)
print("Feature matrix shape:", X_test_shap.shape)

# --- 1. Bar plot ---
print("\nTop 15 SHAP features (CatBoost, reduced model):")
shap.summary_plot(shap_values_cat, X_test_shap, plot_type="bar", max_display=15)

# --- 2. Beeswarm plot ---
shap.summary_plot(shap_values_cat, X_test_shap, max_display=15)

# --- 3. Dependency plots (top 3 features) ---
shap_importance_cat = pd.DataFrame({
    "Feature": X_test_shap.columns,
    "MeanAbsSHAP": np.abs(shap_values_cat).mean(axis=0)
})

top3_cat = shap_importance_cat.sort_values(by="MeanAbsSHAP", ascending=False).head(3)["Feature"].tolist()
print("\nTop 3 features for dependency plots:", top3_cat)

for feat in top3_cat:
    shap.dependence_plot(feat, shap_values_cat, X_test_shap)

# ======================== 10. Feature Importance Across Models ======================== #
print("# ======================== 10. Feature Importance Across Models ======================== #")

# 1. LASSO importance (absolute coefficients)
lasso_importance = pd.Series(
    np.abs(lasso_red.coef_[0]),
    index=X_train_red.columns
).sort_values(ascending=False)

print("\nTop 15 LASSO features:")
print(lasso_importance.head(15))

# 2. Random Forest importance
rf_importance = pd.Series(
    rf_red.feature_importances_,
    index=X_train_red.columns
).sort_values(ascending=False)

print("\nTop 15 Random Forest features:")
print(rf_importance.head(15))

# 3. CatBoost importance (use SHAP importance already computed)
cat_importance = pd.Series(
    np.abs(shap_values_cat).mean(axis=0),
    index=X_test_red.columns
).sort_values(ascending=False)

print("\nTop 15 CatBoost features:")
print(cat_importance.head(15))

# 4. Combined table: top features across models
combined_importance = pd.DataFrame({
    "LASSO": lasso_importance,
    "RandomForest": rf_importance,
    "CatBoost": cat_importance
})

# Normalize ranks for comparison
combined_importance["LASSO_Rank"] = combined_importance["LASSO"].rank(ascending=False)
combined_importance["RF_Rank"] = combined_importance["RandomForest"].rank(ascending=False)
combined_importance["Cat_Rank"] = combined_importance["CatBoost"].rank(ascending=False)

combined_importance["MeanRank"] = combined_importance[["LASSO_Rank","RF_Rank","Cat_Rank"]].mean(axis=1)

combined_sorted = combined_importance.sort_values("MeanRank").head(15)

print("\nTop 15 features across models:")
print(combined_sorted[["LASSO","RandomForest","CatBoost","MeanRank"]])



# ======================== 11. Robust Evaluation of Final Models ======================== #
print("# ======================== 11. Robust Evaluation of Final Models ======================== #")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils import resample
import numpy as np
import pandas as pd

# Final models and their feature matrices
final_models = {
    "LASSO (reduced)": (lasso_red, X_train_red_s, X_test_red_s),
    "Random Forest (reduced)": (rf_red, X_train_red, X_test_red),
    "CatBoost (reduced)": (cat_red, X_train_red, X_test_red)
}

# ---------- 11A. K-fold CV Performance Estimates ----------
print("\n--- 11A. K-fold Cross-Validation Performance ---")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}

for name, (model, Xtr, Xte) in final_models.items():
    auc_scores = cross_val_score(
        model, Xtr, y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )
    cv_results[name] = auc_scores
    print(f"{name}: Mean AUC = {auc_scores.mean():.3f} | Std = {auc_scores.std():.3f} | 95% CI = [{auc_scores.mean() - 1.96*auc_scores.std():.3f}, {auc_scores.mean() + 1.96*auc_scores.std():.3f}]")

# ---------- 11B. Bootstrap Confidence Intervals for Test AUC ----------
print("\n--- 11B. Bootstrapped 95% Confidence Intervals for Test Set AUC ---")

def bootstrap_auc(model, Xtest, ytest, n_boot=2000):
    auc_scores = []
    for i in range(n_boot):
        X_bs, y_bs = resample(Xtest, ytest, replace=True)
        y_prob = model.predict_proba(X_bs)[:, 1]
        auc_scores.append(roc_auc_score(y_bs, y_prob))
    return np.percentile(auc_scores, [2.5, 97.5])

for name, (model, Xtr, Xte) in final_models.items():
    ci_low, ci_high = bootstrap_auc(model, Xte, y_test)
    print(f"{name}: 95% Bootstrap CI for AUC = [{ci_low:.3f}, {ci_high:.3f}]")

# ---------- 11C. Effect Sizes Between Models ----------
print("\n--- 11C. Effect Size (ΔAUC) Between Models ---")

def auc_effect_size(model_a, Xa, model_b, Xb, ytest):
    auc_a = roc_auc_score(ytest, model_a.predict_proba(Xa)[:, 1])
    auc_b = roc_auc_score(ytest, model_b.predict_proba(Xb)[:, 1])
    return auc_a - auc_b

model_names = list(final_models.keys())

for i in range(len(model_names)):
    for j in range(i+1, len(model_names)):
        m1, (mod1, X1, _) = model_names[i], final_models[model_names[i]]
        m2, (mod2, X2, _) = model_names[j], final_models[model_names[j]]
        delta = auc_effect_size(final_models[m1][0], final_models[m1][2],
                                final_models[m2][0], final_models[m2][2],
                                y_test)
        print(f"ΔAUC ({m1} - {m2}) = {delta:.3f}")

# ---------- 11D. DeLong Significance Test ----------
print("\n--- 11D. Statistical Significance: DeLong Test for AUC Differences ---")

import numpy as np
from scipy.stats import norm

def compute_midrank(x):
    """
    Computes midranks for a 1D array.
    Used internally by the DeLong test.
    """
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    N = x.shape[0]
    t = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and sorted_x[j] == sorted_x[i]:
            j += 1
        # Average rank of tied values
        t[i:j] = 0.5 * (i + j - 1)
        i = j
    out = np.empty(N, dtype=float)
    out[sorted_idx] = t + 1
    return out


def compute_midrank_weighted(x, w):
    """
    Weighted midranks. Unused in DeLong for standard ROC AUC.
    Included for completeness.
    """
    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    sorted_w = w[sorted_idx]
    N = x.shape[0]
    t = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and sorted_x[j] == sorted_x[i]:
            j += 1
        # Weighted midranks
        t[i:j] = (np.sum(sorted_w[i:j]) * 0.5 +
                  np.sum(sorted_w[0:i]))
        i = j
    out = np.empty(N, dtype=float)
    out[sorted_idx] = t
    return out


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong implementation for computing AUC variance.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_predictions = predictions_sorted_transposed[:, :m]
    negative_predictions = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.zeros((k, m), dtype=float)
    ty = np.zeros((k, n), dtype=float)

    for r in range(k):
        tx[r, :] = compute_midrank(positive_predictions[r, :])
        ty[r, :] = compute_midrank(negative_predictions[r, :])

    aucs = (tx.sum(axis=1) - m * (m + 1) / 2) / (m * n)

    v10 = (ty - ty.mean(axis=1, keepdims=True)).dot(
        (ty - ty.mean(axis=1, keepdims=True)).T) / n
    v01 = (tx - tx.mean(axis=1, keepdims=True)).dot(
        (tx - tx.mean(axis=1, keepdims=True)).T) / m

    s = v10 / n + v01 / m

    return aucs, s


def calc_pvalue(aucs, sigma):
    """
    Computes two-sided p-value for DeLong AUC difference.
    """
    diff = aucs[0] - aucs[1]
    var = sigma[0, 0] + sigma[1, 1] - 2 * sigma[0, 1]
    z = diff / np.sqrt(var)
    return 2 * (1 - norm.cdf(abs(z)))


def delong_roc_test(y_true, y_scores_1, y_scores_2):
    """
    Performs DeLong test for two correlated ROC AUCs.
    Returns a two-sided p-value.
    """
    y_true = np.array(y_true)
    order = np.argsort(-y_scores_1)

    y_true_sorted = y_true[order]
    scores1_sorted = y_scores_1[order]
    scores2_sorted = y_scores_2[order]

    predictions_sorted_transposed = np.vstack((scores1_sorted, scores2_sorted))
    label_1_count = np.sum(y_true_sorted == 1)

    aucs, sigma = fastDeLong(predictions_sorted_transposed, label_1_count)
    p_value = calc_pvalue(aucs, sigma)
    return p_value


# Run DeLong test for each pair of final models
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        modelA_name = model_names[i]
        modelB_name = model_names[j]

        modelA, XtrA, XteA = final_models[modelA_name]
        modelB, XtrB, XteB = final_models[modelB_name]

        scoresA = modelA.predict_proba(XteA)[:, 1]
        scoresB = modelB.predict_proba(XteB)[:, 1]

        p = delong_roc_test(y_test, scoresA, scoresB)

        print(f"DeLong p-value for {modelA_name} vs {modelB_name}: {p:.5f}")


# ======================== 12. Fairness & Disparate Impact Analysis ======================== #
print("# ======================== 12. Fairness & Disparate Impact Analysis ======================== #")

# Build fairness dataframe directly from the TEST SET
df_fair = df_clean.loc[X_test_red.index, ["BIO_SEX3", "CALCAGE3", "H3ID15"]].copy()

# Create age groups
df_fair["AgeGroup"] = pd.cut(
    df_fair["CALCAGE3"],
    bins=[18, 21, 24, 28],
    labels=["18–21", "21–24", "24–28"],
    include_lowest=True
)

# Add true and predicted values
df_fair["y_true"] = y_test.values
df_fair["y_pred"] = cat_red.predict(X_test_red)
df_fair["y_prob"] = cat_red.predict_proba(X_test_red)[:, 1]

print(f"Fairness dataframe shape: {df_fair.shape}")


# ---------- 12A. Error Rates by Sex ----------
print("\n--- 12A. Error Rates by Sex ---")

def compute_error_rates(g):
    # Confusion components
    TP = ((g["y_pred"] == 1) & (g["y_true"] == 1)).sum()
    TN = ((g["y_pred"] == 0) & (g["y_true"] == 0)).sum()
    FP = ((g["y_pred"] == 1) & (g["y_true"] == 0)).sum()
    FN = ((g["y_pred"] == 0) & (g["y_true"] == 1)).sum()

    # True positives / true negatives count
    P = (g["y_true"] == 1).sum()
    N = (g["y_true"] == 0).sum()

    # Standard definitions
    FNR = FN / P if P > 0 else np.nan
    FPR = FP / N if N > 0 else np.nan
    ACC = (TP + TN) / (P + N)

    return pd.Series({
        "Accuracy": ACC,
        "FNR": FNR,
        "FPR": FPR
    })

sex_error_df = df_fair.groupby("BIO_SEX3").apply(compute_error_rates)
print(sex_error_df)

# --- 12A.1. Confusion matrix counts (CatBoost) by sex ---

print("\n--- 12A.1. Overall CatBoost confusion counts (sanity check) ---")

# Overall confusion components for the CatBoost reduced model on the test set
TN_total = ((df_fair["y_pred"] == 0) & (df_fair["y_true"] == 0)).sum()
FP_total = ((df_fair["y_pred"] == 1) & (df_fair["y_true"] == 0)).sum()
FN_total = ((df_fair["y_pred"] == 0) & (df_fair["y_true"] == 1)).sum()
TP_total = ((df_fair["y_pred"] == 1) & (df_fair["y_true"] == 1)).sum()

print(f"Total TN = {TN_total}, FP = {FP_total}, FN = {FN_total}, TP = {TP_total}")
print("These counts should match the CatBoost confusion matrix (e.g., 714, 152, 30, 79).")

print("\n--- 12A.2. CatBoost confusion counts by sex (BIO_SEX3) ---")

def confusion_counts_by_group(g: pd.DataFrame) -> pd.Series:
    """Return TP, TN, FP, FN counts for a subgroup (e.g. one sex)."""
    TN = ((g["y_pred"] == 0) & (g["y_true"] == 0)).sum()
    FP = ((g["y_pred"] == 1) & (g["y_true"] == 0)).sum()
    FN = ((g["y_pred"] == 0) & (g["y_true"] == 1)).sum()
    TP = ((g["y_pred"] == 1) & (g["y_true"] == 1)).sum()
    return pd.Series({"TN": TN, "FP": FP, "FN": FN, "TP": TP})

sex_confusion_df = df_fair.groupby("BIO_SEX3").apply(confusion_counts_by_group)

print("\nConfusion matrix components (TN, FP, FN, TP) by sex:")
print(sex_confusion_df)

# Optional: save to CSV for reporting
sex_confusion_df.to_csv("CatBoost_confusion_by_sex.csv")
print("\nSaved confusion counts by sex to 'CatBoost_confusion_by_sex.csv'.")



# ---------- 12B. Error Rates by Age Group ----------
print("\n--- 12B. Error Rates by Age Group ---")

age_error_df = df_fair.groupby("AgeGroup").apply(compute_error_rates)
print(age_error_df)


# ---------- 12C. Statistical Parity ----------
print("\n--- 12C. Statistical Parity (Demographic Parity) ---")
group_rates = df_fair.groupby("BIO_SEX3")["y_pred"].mean()
print(group_rates)


# ---------- 12D. Disparate Misclassification ----------
print("\n--- 12D. Disparate Misclassification Table ---")
misclass_table = pd.crosstab(df_fair["BIO_SEX3"], df_fair["y_pred"] != df_fair["y_true"])
print(misclass_table)


# ---------- 12E. SHAP Subgroup Analysis ----------
print("\n--- 12E. SHAP Subgroup Analysis ---")

sex_values = df_fair["BIO_SEX3"].unique()

for g in sex_values:
    mask = df_fair["BIO_SEX3"] == g
    subgroup_data = X_test_red.loc[mask, :]
    subgroup_shap = shap_values_cat[mask]

    print(f"\nSHAP summary for sex group {g}:")
    shap.summary_plot(subgroup_shap, subgroup_data, max_display=10)




# ======================== CONFUSION MATRICES & ROC CURVES ======================== #

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# ======================== 13. Confusion Matrices ======================== #
print("# ======================== 13. Confusion Matrices ======================== #")

models = {
    "LASSO (reduced)": (lasso_red, X_test_red_s),
    "Random Forest (reduced)": (rf_red, X_test_red),
    "CatBoost (reduced)": (cat_red, X_test_red),
}

for name, (model, Xtest) in models.items():
    y_pred = model.predict(Xtest)

    cm = confusion_matrix(y_test, y_pred)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix – {name}")
    plt.show()


# ======================== 14. Combined ROC Curve for All Models ======================== #
print("# ======================== 14. Combined ROC Curve for All Models ======================== #")

plt.figure(figsize=(8, 6))

for name, (model, Xtest) in models.items():
    y_prob = model.predict_proba(Xtest)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

# Plot diagonal reference line
plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for All Models (Reduced Feature Set)")
plt.legend()
plt.tight_layout()
plt.show()

# ======================== 15. Top 15 MI Features + SPSS Labels ======================== #
print("# ======================== 15. Top 15 MI Features + SPSS Labels ======================== #")

# Extract top 15 MI features
top15_mi = mi_series.head(15).reset_index()
top15_mi.columns = ["Variable", "MI_score"]

# Add SPSS labels
top15_mi["SPSS_label"] = top15_mi["Variable"].map(
    lambda v: spss_labels.get(v, "Label not found")
)

print("\nTop 15 MI features with SPSS labels:")
print(top15_mi.to_string(index=False))

# Save table
top15_mi.to_csv("top15_MI_with_labels.csv", index=False)
print("\nSaved as top15_MI_with_labels.csv")

# --- Barplot of Top 15 MI Scores ---
(
    top15_mi.sort_values(by="MI_score", ascending=True)
    .plot(
        kind="barh",
        x="SPSS_label",
        y="MI_score",
        legend=False,
        figsize=(10, 6)
    )
)

plt.title("Top 15 Mutual Information Features (with SPSS labels)")
plt.xlabel("Mutual Information Score")
plt.ylabel("SPSS Variable Label")
plt.tight_layout()
plt.show()




# ======================== 16. Barplot of Tuned Model Performance ======================== #
print("# ======================== 16. Barplot of Tuned Model Performance ======================== #")
plt.figure(figsize=(8,5))
sns.barplot(
    data=tuned_df,
    x="Model",
    y="AUC_Test"
)
plt.title("Tuned Model AUC Comparison")
plt.ylim(0.80, 0.90)
plt.tight_layout()
plt.show()

# ======================== 17. Cross-validation AUC boxplot ======================== #
print("# ======================== 17. Cross-validation AUC boxplot ======================== #")
cv_df = pd.DataFrame({
    "LASSO": cv_results["LASSO (reduced)"],
    "Random Forest": cv_results["Random Forest (reduced)"],
    "CatBoost": cv_results["CatBoost (reduced)"]
})

plt.figure(figsize=(8, 6))
sns.boxplot(data=cv_df)
plt.ylabel("AUC")
plt.title("10-fold Cross-Validation AUC Distribution")
plt.show()

# ======================== 18. Bootstrap 95% CI Forest Plot ======================== #
print("# ======================== 18. Bootstrap 95% CI Forest Plot ======================== #")
import matplotlib.pyplot as plt

models_ci = {
    "LASSO": bootstrap_auc(lasso_red, X_test_red_s, y_test),
    "RF": bootstrap_auc(rf_red, X_test_red, y_test),
    "CatBoost": bootstrap_auc(cat_red, X_test_red, y_test),
}

plt.figure(figsize=(7,5))
for i,(name,ci) in enumerate(models_ci.items()):
    plt.plot(ci, [i,i], lw=5)
    plt.plot([(ci[0]+ci[1])/2], [i], 'o')
plt.yticks(range(len(models_ci)), models_ci.keys())
plt.xlabel("AUC")
plt.title("Bootstrapped 95% Confidence Intervals")
plt.show()

# ======================== 19. Combined Importance Heatmap ======================== #
print("# ======================== 19. Combined Importance Heatmap ======================== #")
importance_rank = combined_importance[["LASSO_Rank","RF_Rank","Cat_Rank"]]

plt.figure(figsize=(10, 8))
sns.heatmap(importance_rank.loc[combined_sorted.index], annot=True, cmap="viridis_r")
plt.title("Feature Importance Rank Across Models")
plt.show()

# ======================== 20. Fairness Barplots ======================== #
print("# ======================== 20. Fairness Barplots ======================== #")
# ======================== 20A. Fairness Barplot: Sex Groups ======================== #
print("\n# ======================== 20A. Fairness Barplot: Sex Groups ======================== #")

sex_df = sex_error_df.reset_index()

plt.figure(figsize=(8,5))

# Plot Accuracy, FNR, FPR side by side
sex_df_melted = sex_df.melt(id_vars="BIO_SEX3", value_vars=["Accuracy", "FNR", "FPR"],
                            var_name="Metric", value_name="Value")

sns.barplot(
    data=sex_df_melted,
    x="BIO_SEX3",
    y="Value",
    hue="Metric"
)

plt.title("Error Rates by Sex")
plt.ylabel("Rate")
plt.xlabel("Sex (BIO_SEX3)")
plt.xticks([0, 1], ["male", "female"])
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# ======================== Fairness Barplot: Age Groups ======================== #
print("\n# ======================== Fairness Barplot: Age Groups ======================== #")

age_error_df.plot(kind="bar", figsize=(8,5))
plt.title("Error Rates by Age Group")
plt.ylabel("Rate")
plt.xlabel("Age Group")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# ======================== 21. Reduced Model Performance Barplot ======================== #
print("# ================= 21. Reduced Model Performance Barplot ================= #")

plt.figure(figsize=(8,5))
sns.barplot(
    data=results_reduced_df,
    x="Model",
    y="AUC_Test"
)
plt.title("Test-Set Performance of Reduced Models (After Removing Depression Indicators)")
plt.ylim(0.75, 0.90)
plt.ylabel("AUC")
plt.xlabel("")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
