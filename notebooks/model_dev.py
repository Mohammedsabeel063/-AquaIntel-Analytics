"""
AquaIntel Analytics — Model Development
Trains:
  1. Full RF-WQI model (all available features) - MAIN
  2. Full XGBoost model
  3. Soft Hybrid (RF + XGB ensemble voting)
Run: python model_dev.py
"""

import os, sys, warnings, joblib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost not installed — skipping XGB model")

from utils.data_loader import (
    load_all_csvs, generate_synthetic_cwc, preprocess,
    CORE_FEATURES
)

from utils.model_utils import SoftVotingHybrid

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = ["#1f4e79", "#2e86ab", "#a23b72", "#f18f01", "#c73e1d"]

# ─── Load & preprocess ───────────────────────────────────────────────────────
try:
    raw = load_all_csvs(DATA_DIR)
except FileNotFoundError:
    print("⚠️  Using synthetic data (no CSVs in /data)")
    raw = generate_synthetic_cwc(n=8000)

df = preprocess(raw)
print(f"Dataset shape: {df.shape}")
print(f"Class balance:\n{df['is_safe'].value_counts(normalize=True).round(3)}\n")

# ─── Feature sets ────────────────────────────────────────────────────────────
all_features  = [f for f in CORE_FEATURES if f in df.columns]

TARGET = "is_safe"

def build_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])

def evaluate_model(pipe, X, y, name, cv=5):
    """Cross-validate and print metrics."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_weighted", "roc_auc"]
    cv_results = cross_validate(pipe, X, y, cv=skf, scoring=scoring,
                                 return_train_score=True, n_jobs=-1)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for metric in scoring:
        tr = cv_results[f"train_{metric}"].mean()
        te = cv_results[f"test_{metric}"].mean()
        te_std = cv_results[f"test_{metric}"].std()
        print(f"  {metric:<18}: train={tr:.4f}  test={te:.4f} ± {te_std:.4f}")
    return cv_results

def plot_confusion_matrix(pipe, X, y, name, outpath):
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Unsafe", "Safe"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Classification report:\n{classification_report(y_te, y_pred, target_names=['Unsafe','Safe'])}")

def plot_feature_importance(pipe, feature_names, name, outpath):
    """Permutation importance."""
    from sklearn.model_selection import train_test_split
    X = df[feature_names].fillna(df[feature_names].median())
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)
    result = permutation_importance(pipe, X_te, y_te, n_repeats=15,
                                    random_state=42, n_jobs=-1)
    imp_mean = result.importances_mean
    imp_std  = result.importances_std
    idx = np.argsort(imp_mean)[::-1][:15]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in idx][::-1],
            imp_mean[idx][::-1], xerr=imp_std[idx][::-1],
            color=PALETTE[1], edgecolor="white", alpha=0.85)
    ax.set_title(f"Permutation Feature Importance — {name}")
    ax.set_xlabel("Mean decrease in accuracy")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

def plot_learning_curve(pipe, X, y, name, outpath):
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, train_scores.mean(1), "o-", color=PALETTE[0], label="Train")
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.15, color=PALETTE[0])
    ax.plot(train_sizes, val_scores.mean(1), "o-", color=PALETTE[2], label="Validation")
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.15, color=PALETTE[2])
    ax.set_xlabel("Training size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve — {name}")
    ax.legend()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# MODEL 1 — Full Random Forest
# ════════════════════════════════════════════════════════════════════════════
rf_full = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=-1)

pipe_rf = build_pipeline(rf_full)

X_full = df[all_features]
y      = df[TARGET]

cv_rf = evaluate_model(pipe_rf, X_full, y, "Random Forest (Full Features)")

plot_confusion_matrix(
    build_pipeline(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=42, n_jobs=-1)),
    X_full, y, "RF Full",
    f"{FIGURES_DIR}/cm_rf_full.png")
print("✅  cm_rf_full.png")

plot_feature_importance(
    build_pipeline(RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=42, n_jobs=-1)),
    all_features, "RF Full",
    f"{FIGURES_DIR}/fi_rf_full.png")
print("✅  fi_rf_full.png")

plot_learning_curve(
    build_pipeline(RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                          random_state=42, n_jobs=-1)),
    X_full, y, "RF Full",
    f"{FIGURES_DIR}/lc_rf_full.png")
print("✅  lc_rf_full.png")

# Save trained model
pipe_rf.fit(X_full, y)
joblib.dump({"model": pipe_rf, "features": all_features, "target": TARGET,
             "type": "full", "algo": "RandomForest"},
            f"{MODELS_DIR}/rf_full.pkl")
print("✅  rf_full.pkl saved")


# ════════════════════════════════════════════════════════════════════════════
# MODEL 2 — XGBoost (Full)
# ════════════════════════════════════════════════════════════════════════════
if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
        random_state=42, n_jobs=-1)

    pipe_xgb = build_pipeline(xgb)
    cv_xgb = evaluate_model(pipe_xgb, X_full, y, "XGBoost (Full Features)")

    plot_confusion_matrix(
        build_pipeline(XGBClassifier(n_estimators=200, max_depth=6,
                                      learning_rate=0.05, use_label_encoder=False,
                                      eval_metric="logloss", random_state=42)),
        X_full, y, "XGB Full",
        f"{FIGURES_DIR}/cm_xgb_full.png")
    print("✅  cm_xgb_full.png")

    pipe_xgb.fit(X_full, y)
    joblib.dump({"model": pipe_xgb, "features": all_features, "target": TARGET,
                 "type": "full", "algo": "XGBoost"},
                f"{MODELS_DIR}/xgb_full.pkl")
    print("✅  xgb_full.pkl saved")


# ════════════════════════════════════════════════════════════════════════════
# MODEL 3 — Soft Hybrid (RF + XGB Ensemble)
# ════════════════════════════════════════════════════════════════════════════
if HAS_XGB:
    # Use trained model pipelines to create hybrid
    hybrid_model = SoftVotingHybrid(pipe_rf, pipe_xgb)
    
    # Evaluate hybrid on test split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.2,
                                               random_state=42, stratify=y)
    hybrid_model.fit(X_tr, y_tr)
    
    # Get predictions and metrics
    y_pred_hybrid = hybrid_model.predict(X_te)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    acc_hybrid = accuracy_score(y_te, y_pred_hybrid)
    f1_hybrid = f1_score(y_te, y_pred_hybrid, average='weighted')
    auc_hybrid = roc_auc_score(y_te, hybrid_model.predict_proba(X_te)[:, 1])
    
    print(f"\n{'='*50}")
    print(f"  Soft Hybrid (RF + XGB)")
    print(f"{'='*50}")
    print(f"  accuracy          : {acc_hybrid:.4f}")
    print(f"  f1_weighted       : {f1_hybrid:.4f}")
    print(f"  roc_auc           : {auc_hybrid:.4f}")
    print(f"  Classification report:\n{classification_report(y_te, y_pred_hybrid, target_names=['Unsafe','Safe'])}")
    
    # Create CV-compatible results for summary
    cv_hybrid = {
        'test_accuracy': np.array([acc_hybrid]),
        'test_f1_weighted': np.array([f1_hybrid]),
        'test_roc_auc': np.array([auc_hybrid]),
    }
    
    # Plot confusion matrix
    cm = confusion_matrix(y_te, y_pred_hybrid)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Unsafe", "Safe"]).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix — Soft Hybrid")
    fig.savefig(f"{FIGURES_DIR}/cm_hybrid_soft.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✅  cm_hybrid_soft.png")
    
    # Save hybrid model
    joblib.dump({"model": hybrid_model, "features": all_features, "target": TARGET,
                 "type": "hybrid", "algo": "SoftVoting(RF+XGB)"},
                f"{MODELS_DIR}/hybrid_soft.pkl")
    print("✅  hybrid_soft.pkl saved")


# ════════════════════════════════════════════════════════════════════════════
# Comparison Summary
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

def extract_cv(cv_results):
    return {
        "Accuracy": f"{cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}",
        "F1":       f"{cv_results['test_f1_weighted'].mean():.4f} ± {cv_results['test_f1_weighted'].std():.4f}",
        "AUC-ROC":  f"{cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}",
        "Features": len(all_features),
    }

rows = [
    {"Model": "RF Full (Main)",  **{k: v for k, v in extract_cv(cv_rf).items()}},
]
if HAS_XGB:
    rows.insert(1, {"Model": "XGB Full", **{k: v for k, v in extract_cv(cv_xgb).items()}})
    if 'cv_hybrid' in locals():
        rows.insert(2, {"Model": "Soft Hybrid", **{k: v for k, v in extract_cv(cv_hybrid).items()}})

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(f"{FIGURES_DIR}/model_comparison.csv", index=False)
print("\n✅  model_comparison.csv saved")
print("✅  Model development complete.")
# Model update: RF (main), XGBoost, and RF+XGB hybrid added; Lite model removed

