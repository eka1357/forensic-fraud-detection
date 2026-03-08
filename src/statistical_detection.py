"""
src/statistical_detection.py
------------------------------
Statistical anomaly detection for forensic financial analysis.

Methods
-------
1. Benford's Law            — first-digit distribution test (Nigrini MAD threshold)
2. Z-Score detection        — parametric outlier flagging
3. IQR / Tukey fences       — robust non-parametric outlier detection
4. Isolation Forest         — unsupervised ML anomaly isolation
5. Peer Group Benchmarking  — sector-level z-score comparison
6. Composite Anomaly Score  — weighted combination of all flags (0–100)

Forensic reference: Nigrini (2012) Benford's Law: Applications for Forensic
Accounting, Auditing, and Fraud Detection. Wiley.
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Theoretical Benford probabilities — Newcomb (1881) / Benford (1938)
BENFORD_EXPECTED = {
    1: 0.30103, 2: 0.17609, 3: 0.12494, 4: 0.09691,
    5: 0.07918, 6: 0.06695, 7: 0.05799, 8: 0.05115, 9: 0.04576,
}

CTR_THRESHOLD     = 10_000.0   # UK SAR reporting threshold
STRUCTURING_LOWER =  9_500.0   # Conservative structuring detection window


# ══════════════════════════════════════════════════════════════════════════════
# 1. Benford's Law
# ══════════════════════════════════════════════════════════════════════════════

def benford_analysis(df: pd.DataFrame, amount_col: str = "amount_gbp") -> dict:
    """
    Test transaction amounts against Benford's Law.

    Returns dict with chi2_stat, chi2_p, MAD, Nigrini conformity rating,
    and list of suspicious digits (|Z| > 2.0).

    Nigrini MAD thresholds (2012):
        < 0.006  → Close Conformity
        < 0.012  → Acceptable Conformity
        < 0.015  → Marginally Acceptable
        ≥ 0.015  → NON-CONFORMITY (investigate)
    """
    amounts = df[amount_col].dropna()
    amounts = amounts[amounts > 0]
    digits  = amounts.apply(_leading_digit).dropna().astype(int)

    obs_counts = digits.value_counts().reindex(range(1, 10), fill_value=0)
    total      = obs_counts.sum()
    obs_freq   = (obs_counts / total).to_dict()

    # Chi-squared goodness-of-fit test
    exp_counts = np.array([BENFORD_EXPECTED[d] * total for d in range(1, 10)])
    obs_arr    = np.array([obs_counts[d] for d in range(1, 10)])
    chi2_stat, chi2_p = stats.chisquare(obs_arr, f_exp=exp_counts)

    # MAD — Mean Absolute Deviation
    mad = np.mean([abs(obs_freq.get(d, 0) - BENFORD_EXPECTED[d]) for d in range(1, 10)])

    # Z-scores per digit
    digit_zscores = {}
    suspicious    = []
    for d in range(1, 10):
        obs = obs_freq.get(d, 0)
        exp = BENFORD_EXPECTED[d]
        se  = (exp * (1 - exp) / total) ** 0.5 if total > 0 else 1
        z   = (obs - exp) / se
        digit_zscores[d] = round(z, 3)
        if z > 2.0:
            suspicious.append(d)

    if   mad < 0.006:  conformity = "Close Conformity"
    elif mad < 0.012:  conformity = "Acceptable Conformity"
    elif mad < 0.015:  conformity = "Marginally Acceptable"
    else:              conformity = "NON-CONFORMITY — INVESTIGATE"

    return {
        "observed_freq":     obs_freq,
        "expected_freq":     BENFORD_EXPECTED,
        "obs_counts":        obs_counts.to_dict(),
        "total_transactions": int(total),
        "chi2_stat":         round(chi2_stat, 4),
        "chi2_p":            round(chi2_p, 6),
        "mad":               round(mad, 6),
        "digit_zscores":     digit_zscores,
        "suspicious_digits": suspicious,
        "nigrini_conformity": conformity,
    }


def _leading_digit(x: float):
    if x <= 0:
        return None
    s = f"{x:.2f}".lstrip("0").replace(".", "")
    return int(s[0]) if s else None


def plot_benford(result: dict, title: str = "Benford's Law Analysis",
                 save_path: str | None = None):
    """Grouped bar chart: observed vs Benford expected first-digit frequency."""
    digits = list(range(1, 10))
    obs    = [result["observed_freq"].get(d, 0) * 100 for d in digits]
    exp    = [result["expected_freq"][d] * 100 for d in digits]
    x, w   = np.arange(len(digits)), 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, obs, w, label="Observed",          color="#1A1A2E", edgecolor="white")
    ax.bar(x + w/2, exp, w, label="Benford Expected",  color="#E74C3C", alpha=0.8, edgecolor="white")

    # Highlight suspicious digits in red x-labels
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    for d in result["suspicious_digits"]:
        ax.get_xticklabels()[d - 1].set_color("red")
        ax.get_xticklabels()[d - 1].set_fontweight("bold")

    ax.set_xlabel("Leading Digit", fontsize=11)
    ax.set_ylabel("Frequency (%)", fontsize=11)
    ax.set_title(
        f"{title}\nMAD = {result['mad']:.4f}  |  {result['nigrini_conformity']}  "
        f"|  χ²-p = {result['chi2_p']:.4f}",
        fontsize=13,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Z-Score Detection
# ══════════════════════════════════════════════════════════════════════════════

def zscore_detection(df: pd.DataFrame, amount_col: str = "amount_gbp",
                     threshold: float = 3.0) -> pd.DataFrame:
    """Flag transactions > threshold standard deviations from mean amount."""
    df = df.copy()
    df["amount_zscore"] = stats.zscore(df[amount_col].fillna(0))
    df["zscore_flag"]   = (df["amount_zscore"].abs() > threshold).astype(int)
    n = df["zscore_flag"].sum()
    logger.info("Z-Score (threshold=%.1f): %d flagged (%.1f%%)", threshold, n, n/len(df)*100)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. IQR / Tukey Fences
# ══════════════════════════════════════════════════════════════════════════════

def iqr_detection(df: pd.DataFrame, amount_col: str = "amount_gbp",
                  k: float = 1.5) -> pd.DataFrame:
    """
    Flag transactions outside the Tukey fence [Q1 − k·IQR, Q3 + k·IQR].
    k=1.5 → inner fence; k=3.0 → outer fence (extreme outliers).
    """
    df  = df.copy()
    q1  = df[amount_col].quantile(0.25)
    q3  = df[amount_col].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + k * iqr
    lower = q1 - k * iqr

    df["iqr_upper_fence"] = upper
    df["iqr_lower_fence"] = lower
    df["iqr_flag"]        = ((df[amount_col] > upper) | (df[amount_col] < lower)).astype(int)
    n = df["iqr_flag"].sum()
    logger.info("IQR (k=%.1f): fence [£%.0f, £%.0f] | %d flagged", k, lower, upper, n)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Isolation Forest
# ══════════════════════════════════════════════════════════════════════════════

def isolation_forest_detection(df: pd.DataFrame,
                                feature_cols: list | None = None,
                                contamination: float = 0.10,
                                random_state: int = 42) -> pd.DataFrame:
    """
    Unsupervised anomaly detection using Isolation Forest.
    Higher `if_score` = more anomalous (inverted sign convention).
    """
    df = df.copy()
    if feature_cols is None:
        candidates = ["amount_gbp", "log_amount", "is_pep", "is_cross_border",
                      "is_high_risk_cpty", "is_round_amount", "is_near_ctr"]
        feature_cols = [c for c in candidates if c in df.columns]

    X = StandardScaler().fit_transform(df[feature_cols].fillna(0))
    iso = IsolationForest(contamination=contamination, n_estimators=300,
                          max_samples="auto", random_state=random_state, n_jobs=-1)
    iso.fit(X)

    df["if_score"] = -iso.score_samples(X)      # higher = more anomalous
    df["if_flag"]  = (iso.predict(X) == -1).astype(int)
    n = df["if_flag"].sum()
    logger.info("Isolation Forest (contamination=%.2f): %d flagged", contamination, n)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. Peer Group Analysis
# ══════════════════════════════════════════════════════════════════════════════

def peer_group_analysis(df: pd.DataFrame, group_col: str = "sector",
                        amount_col: str = "amount_gbp",
                        z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag transactions where the amount is an outlier relative to sector peers.
    Captures cases where legitimate-looking absolute amounts are anomalous
    for their industry context.
    """
    df = df.copy()
    stats_df = df.groupby(group_col)[amount_col].agg(["mean","std"]).rename(
        columns={"mean": "peer_mean", "std": "peer_std"}
    )
    df = df.join(stats_df, on=group_col)
    df["peer_zscore"] = (
        (df[amount_col] - df["peer_mean"]) /
        df["peer_std"].replace(0, np.nan)
    ).fillna(0)
    df["peer_flag"] = (df["peer_zscore"].abs() > z_threshold).astype(int)
    n = df["peer_flag"].sum()
    logger.info("Peer Group (%s, z>%.1f): %d flagged", group_col, z_threshold, n)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. Composite Anomaly Score
# ══════════════════════════════════════════════════════════════════════════════

STAT_FLAG_WEIGHTS = {
    "zscore_flag": 20,
    "iqr_flag":    15,
    "if_flag":     30,
    "peer_flag":   20,
    "is_near_ctr": 15,   # structuring signal
}

def build_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted composite of statistical flags → 0–100 anomaly score.

    Risk tiers:
        0–20   Low
        21–40  Medium
        41–65  High
        66–100 Critical
    """
    df = df.copy()
    available = {c: w for c, w in STAT_FLAG_WEIGHTS.items() if c in df.columns}
    total_w   = sum(available.values())

    df["stat_composite"] = sum(
        df[c].fillna(0) * (w / total_w * 100)
        for c, w in available.items()
    ).round(1)

    df["stat_risk_tier"] = pd.cut(
        df["stat_composite"],
        bins=[-1, 20, 40, 65, 101],
        labels=["Low", "Medium", "High", "Critical"],
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def plot_amount_distribution(df: pd.DataFrame, save_path: str | None = None):
    """Log-scale histogram of amounts — fraud overlay with CTR threshold line."""
    fig, ax = plt.subplots(figsize=(11, 5))
    bins = np.logspace(
        np.log10(max(df["amount_gbp"].min(), 10)),
        np.log10(df["amount_gbp"].max()), 55
    )
    ax.hist(df[df["is_fraud"]==0]["amount_gbp"], bins=bins, alpha=0.65,
            color="#1A1A2E", label="Legitimate", density=True)
    ax.hist(df[df["is_fraud"]==1]["amount_gbp"], bins=bins, alpha=0.75,
            color="#E74C3C", label="Fraudulent",  density=True)
    ax.axvline(CTR_THRESHOLD, color="#F39C12", linewidth=2, linestyle="--",
               label=f"CTR Threshold £{CTR_THRESHOLD:,.0f}")
    ax.axvspan(STRUCTURING_LOWER, CTR_THRESHOLD, alpha=0.08, color="#E74C3C",
               label="Structuring Window")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.set_xlabel("Transaction Amount (log scale)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Transaction Amount Distribution — Fraud vs Legitimate", fontsize=14)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_fraud_by_type(df: pd.DataFrame, save_path: str | None = None):
    """Horizontal bar chart of fraud counts by type."""
    fraud_df = df[df["is_fraud"] == 1]
    if fraud_df.empty: return
    counts = fraud_df["fraud_type"].value_counts()
    pcts   = counts / len(df) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#922B21","#C0392B","#E74C3C","#F1948A","#FADBD8","#F39C12"]
    bars = counts.sort_values().plot(kind="barh", ax=ax,
                                      color=colors[:len(counts)][::-1], edgecolor="white")
    for i, (v, p) in enumerate(zip(counts.sort_values(), pcts[counts.sort_values().index])):
        ax.text(v + 1, i, f"  {p:.1f}%", va="center", fontsize=9, color="#555")
    ax.set_title("Fraudulent Transactions by Type", fontsize=14)
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_structuring_zoom(df: pd.DataFrame, save_path: str | None = None):
    """Zoomed histogram around the CTR threshold to highlight structuring."""
    window = df[df["amount_gbp"].between(8000, 11000)]
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.linspace(8000, 11000, 60)
    ax.hist(window[window["is_fraud"]==0]["amount_gbp"], bins=bins, alpha=0.7,
            color="#1A1A2E", label="Legitimate")
    ax.hist(window[window["is_fraud"]==1]["amount_gbp"], bins=bins, alpha=0.8,
            color="#E74C3C", label="Fraudulent")
    ax.axvline(CTR_THRESHOLD, color="#F39C12", linewidth=2.5, linestyle="--",
               label="CTR Threshold £10,000")
    ax.axvspan(STRUCTURING_LOWER, CTR_THRESHOLD, alpha=0.12, color="#E74C3C")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.set_xlabel("Transaction Amount"); ax.set_ylabel("Count")
    ax.set_title("Structuring Window Analysis (£8k–£11k)", fontsize=13)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved → '%s'", save_path)
    else:
        plt.show()
    plt.close(fig)
