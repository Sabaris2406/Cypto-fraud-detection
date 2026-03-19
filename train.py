"""
=============================================================
 CRYPTOCURRENCY FRAUD DETECTION SYSTEM
 Module 3, 4 & 5 — Preprocessing, Feature Engineering,
                    Model Training & Evaluation
 Dataset : Elliptic Bitcoin Dataset
 Author  : Krishna Kumar S (23CTU033)
 Guide   : Mrs. R. Poongodi
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score
)
import xgboost as xgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

# ── Plot style ────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0F1923",
    "axes.facecolor":   "#1A2535",
    "axes.edgecolor":   "#2E3F55",
    "text.color":       "#E0E8F0",
    "axes.labelcolor":  "#E0E8F0",
    "xtick.color":      "#8AA0B8",
    "ytick.color":      "#8AA0B8",
    "grid.color":       "#2E3F55",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})

ILLICIT_COLOR = "#EF4444"
LICIT_COLOR   = "#22C55E"
ACCENT_COLOR  = "#3B82F6"

OUTPUT_DIR = "training_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20


def engineer_features(df: pd.DataFrame):
    """Add 3 engineered features to the raw labelled DataFrame."""
    print("[Feature Engineering] Creating new features …")
    labelled = df[df["class"] != "unknown"].copy()

    if "f3" in labelled.columns and "f2" in labelled.columns:
        labelled["fee_ratio"] = labelled["f3"] / (labelled["f2"] + 1e-9)

    if "f4" in labelled.columns and "f5" in labelled.columns:
        labelled["io_ratio"] = labelled["f4"] / (labelled["f5"] + 1e-9)

    labelled["is_illicit"] = (labelled["class"] == "1").astype(int)
    time_fraud_map = (
        labelled.groupby("time_step")["is_illicit"].mean()
        .rename("time_step_fraud_rate").to_dict()
    )
    labelled["time_step_fraud_rate"] = labelled["time_step"].map(time_fraud_map)
    labelled.drop(columns=["is_illicit"], inplace=True)

    print(f"  New features added: fee_ratio, io_ratio, time_step_fraud_rate")
    return labelled, time_fraud_map


def preprocess(df: pd.DataFrame):
    """Filter labelled rows, encode target, impute, split, scale."""
    print("[Preprocessing] Filtering labelled transactions …")
    labelled = df[df["class"] != "unknown"].copy()
    labelled["target"] = labelled["class"].apply(
        lambda x: 1 if str(x).strip() == "1" else 0)
    print(f"  Labelled rows  : {len(labelled):,}")
    print(f"  Illicit (1)    : {labelled['target'].sum():,}")
    print(f"  Licit   (0)    : {(labelled['target'] == 0).sum():,}")

    feature_cols = [c for c in labelled.columns
                    if c not in ["txId", "class", "target"]]
    X = labelled[feature_cols].copy()
    y = labelled["target"].copy()

    print("[Preprocessing] Imputing missing values …")
    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_arr = imputer.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=feature_cols, index=X.index)

    print("[Preprocessing] Stratified train/test split …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y)
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    print("[Preprocessing] Scaling features …")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test = pd.DataFrame(X_test_scaled, columns=feature_cols)

    return X_train, X_test, y_train, y_test, scaler, imputer, feature_cols


def select_features(X_train: pd.DataFrame,
                    X_test: pd.DataFrame,
                    var_threshold: float = 0.01,
                    corr_threshold: float = 0.95):
    """Remove near-zero variance and highly correlated features."""
    print("[Feature Selection] Variance threshold …")
    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(X_train)
    kept = X_train.columns[vt.get_support()].tolist()
    X_train = X_train[kept]
    X_test  = X_test[kept]
    print(f"  After variance threshold: {len(kept)} features")

    print("[Feature Selection] Correlation-based filtering …")
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns
               if any(upper[col] > corr_threshold)]
    X_train.drop(columns=to_drop, inplace=True, errors="ignore")
    X_test.drop(columns=to_drop,  inplace=True, errors="ignore")
    final_features = X_train.columns.tolist()
    print(f"  Dropped {len(to_drop)} highly correlated features")
    print(f"  Final feature count: {len(final_features)}\n")

    return X_train, X_test, final_features


def train_baseline(X_train, y_train, X_val, y_val, scale_weight):
    """Train a baseline XGBoost model."""
    print("[Training] Baseline XGBoost …")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    print(f"  Training complete")
    return model


def tune_hyperparameters(X_train, y_train, X_val, y_val, scale_weight, n_trials=50):
    """Bayesian hyperparameter tuning with Optuna."""
    print(f"[Tuning] Optuna — {n_trials} trials …")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "scale_pos_weight": scale_weight,
            "eval_metric":      "aucpr",
            "random_state":     RANDOM_STATE,
            "n_jobs":           -1,
            "verbosity":        0,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train)
        proba = m.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, proba)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    print(f"  Best PR-AUC: {study.best_value:.4f}")
    print(f"  Best params: {best}")
    return best


def train_final_model(X_train, y_train, X_val, y_val, best_params, scale_weight):
    """Train final model with best hyperparameters."""
    print("[Training] Final model with best params …")
    best_params["scale_pos_weight"] = scale_weight
    best_params["eval_metric"]      = "aucpr"
    best_params["random_state"]     = RANDOM_STATE
    best_params["n_jobs"]           = -1
    best_params["verbosity"]        = 0

    final = xgb.XGBClassifier(**best_params)
    final.fit(X_train, y_train)
    print(f"  Training complete")
    return final


def evaluate_model(model, X_test, y_test):
    """Full evaluation: report, confusion matrix, ROC, PR curve."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred,
                                target_names=["Licit", "Illicit"]))
    print(f"AUC-ROC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, y_prob):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Model Evaluation — Cryptocurrency Fraud Detection",
                 color="#E0E8F0", fontsize=15, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=["Licit", "Illicit"],
                yticklabels=["Licit", "Illicit"],
                ax=axes[0], linewidths=0.5,
                annot_kws={"size": 14, "color": "black"})
    axes[0].set_title("Confusion Matrix", color="#E0E8F0", fontsize=13)
    axes[0].set_xlabel("Predicted", color="#E0E8F0")
    axes[0].set_ylabel("Actual", color="#E0E8F0")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    axes[1].plot(fpr, tpr, color=ACCENT_COLOR, lw=2,
                 label=f"AUC = {auc_val:.4f}")
    axes[1].plot([0, 1], [0, 1], color="#8AA0B8", lw=1.2,
                 linestyle="--", label="Random")
    axes[1].fill_between(fpr, tpr, alpha=0.15, color=ACCENT_COLOR)
    axes[1].set_title("ROC Curve", color="#E0E8F0", fontsize=13)
    axes[1].set_xlabel("False Positive Rate", color="#E0E8F0")
    axes[1].set_ylabel("True Positive Rate", color="#E0E8F0")
    axes[1].legend(fontsize=10)
    axes[1].grid(True)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    axes[2].plot(recall, precision, color=ILLICIT_COLOR, lw=2,
                 label=f"PR-AUC = {pr_auc:.4f}")
    axes[2].fill_between(recall, precision, alpha=0.15, color=ILLICIT_COLOR)
    axes[2].set_title("Precision-Recall Curve", color="#E0E8F0", fontsize=13)
    axes[2].set_xlabel("Recall", color="#E0E8F0")
    axes[2].set_ylabel("Precision", color="#E0E8F0")
    axes[2].legend(fontsize=10)
    axes[2].grid(True)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_evaluation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1923")
    print(f"[Evaluation] Saved → {path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plot XGBoost built-in feature importance."""
    imp = model.get_booster().get_fscore()
    imp_df = pd.DataFrame(list(imp.items()),
                          columns=["Feature", "Importance"])
    imp_df.sort_values("Importance", ascending=False, inplace=True)
    top = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1],
            color=ACCENT_COLOR, edgecolor="#0F1923", linewidth=0.8)
    ax.set_title(f"XGBoost Feature Importance — Top {top_n}",
                 color="#E0E8F0", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance (Gain)", color="#E0E8F0")
    ax.grid(axis="x")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "07_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1923")
    print(f"[Importance] Saved → {path}")
    plt.close()


def generate_shap_plots(model, X_test, max_display=20, sample_size=500):
    """Generate SHAP summary plots."""
    print("[SHAP] Computing SHAP values …")
    sample = X_test.sample(min(sample_size, len(X_test)),
                           random_state=RANDOM_STATE)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, sample,
                      plot_type="bar",
                      max_display=max_display,
                      show=False, color=ACCENT_COLOR)
    plt.title("SHAP Global Feature Importance",
              color="#E0E8F0", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "08_shap_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1923")
    print(f"[SHAP] Saved → {path}")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, sample,
                      max_display=max_display,
                      show=False)
    plt.title("SHAP Beeswarm — Feature Impact",
              color="#E0E8F0", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "09_shap_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1923")
    print(f"[SHAP] Saved → {path}")
    plt.close()


def save_artifacts(model, scaler, feature_names, imputer=None):
    """Save model, scaler, and feature names to disk."""
    model.save_model("crypto_fraud_xgboost.json")
    print("[Artifacts] Model saved → crypto_fraud_xgboost.json")

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[Artifacts] Scaler saved → scaler.pkl")

    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print(f"[Artifacts] {len(feature_names)} feature names saved → feature_names.pkl")

    if imputer is not None:
        with open("imputer.pkl", "wb") as f:
            pickle.dump(imputer, f)
        print("[Artifacts] Imputer saved → imputer.pkl")

    test_model = xgb.XGBClassifier()
    test_model.load_model("crypto_fraud_xgboost.json")
    print(f"[Artifacts] Verification OK — {test_model.n_estimators} estimators")


def run_training_pipeline(features_path="elliptic_txs_features.csv",
                          classes_path="elliptic_txs_classes.csv",
                          tune=True,
                          n_trials=50):
    """End-to-end training pipeline."""

    df = pd.read_csv(features_path, header=None)
    df.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, 166)]
    classes = pd.read_csv(classes_path)
    classes.columns = classes.columns.str.strip()
    df = df.merge(classes, on="txId", how="left")

    df, time_fraud_map = engineer_features(df)

    (X_train, X_test,
     y_train, y_test,
     scaler, imputer, feature_cols) = preprocess(df)

    X_train, X_test, final_features = select_features(X_train, X_test)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    scale_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"[Training] scale_pos_weight = {scale_weight:.2f}\n")

    model_baseline = train_baseline(X_train, y_train,
                                    X_test,  y_test,
                                    scale_weight)

    if tune:
        best_params = tune_hyperparameters(X_train, y_train,
                                           X_test,  y_test,
                                           scale_weight, n_trials)
    else:
        best_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0.0,
        }

    final_model = train_final_model(X_train, y_train,
                                    X_test,  y_test,
                                    best_params, scale_weight)

    print("\n--- BASELINE MODEL ---")
    evaluate_model(model_baseline, X_test, y_test)
    print("\n--- FINAL TUNED MODEL ---")
    evaluate_model(final_model, X_test, y_test)

    plot_feature_importance(final_model, final_features)
    generate_shap_plots(final_model, X_test)
    save_artifacts(final_model, scaler, final_features, imputer)

    print("\n[Pipeline] Training complete! All outputs in:", OUTPUT_DIR)
    return final_model


if __name__ == "__main__":
    run_training_pipeline(tune=True, n_trials=30)
