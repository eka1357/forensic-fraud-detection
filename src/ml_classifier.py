"""
src/ml_classifier.py
---------------------
Supervised fraud classification layer combining statistical and rule-based
features with raw transaction attributes.

Models
------
Random Forest   — robust, handles non-linear feature interactions
XGBoost         — best tabular performance; handles class imbalance via scale_pos_weight

Feature groups
--------------
A. Raw transaction    amount, type, entity, sector, timing
B. Statistical flags  z-score, IQR, Isolation Forest, peer group
C. Rule-based flags   R01–R10 compliance rules
D. Derived            log_amount, is_weekend, is_cross_border, etc.
"""

import os
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, RandomizedSearchCV,
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_score, recall_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

logger      = logging.getLogger(__name__)
RANDOM_STATE= 42
TARGET      = "is_fraud"
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")

# ── Feature column lists ───────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "amount_gbp", "log_amount", "is_pep",
    "amount_zscore", "if_score", "peer_zscore",
    "rule_flag_count", "rule_score",
]

CATEGORICAL_FEATURES = [
    "entity_type", "sector", "transaction_type",
    "risk_rating", "counterparty_country", "purpose",
]

BINARY_FEATURES = [
    "is_weekend", "is_cross_border", "is_high_risk_cpty",
    "is_round_amount", "is_near_ctr", "is_unknown_purpose",
    "R01_CTR_PROXIMITY", "R02_ROUND_AMOUNT", "R03_HIGH_RISK_COUNTRY",
    "R04_PEP_HIGH_VALUE", "R05_CROSS_BORDER_LARGE", "R06_UNKNOWN_PURPOSE",
    "R07_WEEKEND_WIRE",   "R08_HIGH_RISK_ACCOUNT",  "R09_OPAQUE_ENTITY",
    "R10_FX_LARGE",
]

CLASSIFIERS = {
    "random_forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", max_depth=None,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=5,           # compensate for ~16% fraud rate
        eval_metric="aucpr",
        random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
    ),
}

PARAM_GRIDS = {
    "random_forest": {
        "classifier__n_estimators":      [100, 200, 300],
        "classifier__max_depth":         [None, 5, 10, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf":  [1, 2, 4],
    },
    "xgboost": {
        "classifier__n_estimators":       [200, 300, 500],
        "classifier__max_depth":          [3, 5, 7],
        "classifier__learning_rate":      [0.01, 0.05, 0.1],
        "classifier__subsample":          [0.7, 0.8, 1.0],
        "classifier__scale_pos_weight":   [3, 5, 8],
    },
}


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in NUMERIC_FEATURES   if c in X.columns]
    cat = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    bin = [c for c in BINARY_FEATURES    if c in X.columns]

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    bin_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0))])

    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", cat_pipe, cat),
        ("bin", bin_pipe, bin),
    ], remainder="drop", verbose_feature_names_out=True)


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, test_size: float = 0.20) -> tuple:
    DROP = [TARGET, "transaction_id", "date", "time", "account_id", "account_holder",
            "counterparty_id", "reference", "fraud_type", "triggered_rules",
            "rule_risk_tier", "stat_risk_tier", "if_flag", "zscore_flag", "iqr_flag",
            "peer_flag", "iqr_upper_fence", "iqr_lower_fence",
            "peer_mean", "peer_std", "originating_bank", "counterparty_bank",
            "account_holder", "leading_digit"]
    X = df.drop(columns=[c for c in DROP if c in df.columns])
    y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("Split → train:%d test:%d | fraud in train: %.1f%%",
                len(y_tr), len(y_te), y_tr.mean()*100)
    return X_tr, X_te, y_tr, y_te


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(name: str, X_tr, y_tr, tune: bool = False, n_iter: int = 15) -> Pipeline:
    pre  = build_preprocessor(X_tr)
    clf  = CLASSIFIERS[name]
    pipe = Pipeline([("preprocessor", pre), ("classifier", clf)])

    if tune and name in PARAM_GRIDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        rs = RandomizedSearchCV(pipe, PARAM_GRIDS[name], n_iter=n_iter,
                                scoring="average_precision", cv=cv,
                                n_jobs=-1, random_state=RANDOM_STATE, verbose=1)
        rs.fit(X_tr, y_tr)
        logger.info("%s best CV AP: %.4f | params: %s", name, rs.best_score_, rs.best_params_)
        return rs.best_estimator_

    pipe.fit(X_tr, y_tr)
    logger.info("%s trained (no tuning).", name)
    return pipe


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(name: str, pipe: Pipeline, X_te, y_te) -> dict:
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "model":         name,
        "accuracy":      round((tp + tn) / (tp + tn + fp + fn), 4),
        "precision":     round(precision_score(y_te, y_pred, zero_division=0), 4),
        "recall":        round(recall_score(y_te, y_pred, zero_division=0), 4),
        "f1":            round(f1_score(y_te, y_pred, zero_division=0), 4),
        "specificity":   round(spec, 4),
        "roc_auc":       round(roc_auc_score(y_te, y_prob), 4),
        "avg_precision": round(average_precision_score(y_te, y_prob), 4),
        "report":        classification_report(y_te, y_pred, target_names=["Legitimate","Fraud"]),
        "y_pred":        y_pred,
        "y_prob":        y_prob,
    }


def compare_models(results: list) -> pd.DataFrame:
    cols = ["model","accuracy","precision","recall","f1","specificity","roc_auc","avg_precision"]
    return (pd.DataFrame([{c: r[c] for c in cols} for r in results])
              .sort_values("avg_precision", ascending=False).reset_index(drop=True))


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(pipe: Pipeline, name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(pipe, path)
    logger.info("Model saved → '%s'", path)
    return path


def load_model(name: str) -> Pipeline:
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.joblib"))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_roc_pr_curves(model_results: dict, save_path: str | None = None):
    """Side-by-side ROC and PR curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#E74C3C","#1A1A2E","#27AE60","#F39C12"]

    for i, (name, res) in enumerate(model_results.items()):
        c = colors[i % len(colors)]
        RocCurveDisplay.from_predictions(res["y_true"], res["y_prob"],
                                          name=f"{name} (AUC={res['roc_auc']:.3f})",
                                          ax=axes[0], color=c)
        PrecisionRecallDisplay.from_predictions(res["y_true"], res["y_prob"],
                                                 name=f"{name} (AP={res['avg_precision']:.3f})",
                                                 ax=axes[1], color=c)

    axes[0].plot([0,1],[0,1],"k--",lw=1)
    axes[0].set_title("ROC Curves", fontsize=13)
    axes[1].set_title("Precision-Recall Curves", fontsize=13)
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle("Model Comparison — Test Set Performance", fontsize=14)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else: plt.show()
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, model_name: str = "",
                           save_path: str | None = None):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Legitimate","Fraud"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else: plt.show()
    plt.close(fig)


def plot_feature_importance(pipe: Pipeline, top_n: int = 25,
                             title: str = "", save_path: str | None = None):
    clf  = pipe.named_steps["classifier"]
    prep = pipe.named_steps["preprocessor"]

    if hasattr(clf, "feature_importances_"):
        imps = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imps = np.abs(clf.coef_[0])
    else:
        return

    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = [f"f{i}" for i in range(len(imps))]

    n = min(len(imps), len(names))
    feat = pd.Series(imps[:n], index=names[:n]).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#E74C3C" if i < 5 else "#95A5A6" for i in range(len(feat))]
    feat.sort_values().plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
    ax.set_title(f"Feature Importances{' — '+title if title else ''}", fontsize=14)
    ax.set_xlabel("Importance Score")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else: plt.show()
    plt.close(fig)
