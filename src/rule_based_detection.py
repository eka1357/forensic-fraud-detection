"""
src/rule_based_detection.py
----------------------------
Deterministic compliance rule engine for AML / financial crime detection.

Regulatory framework
---------------------
- POCA 2002      Proceeds of Crime Act — s.330 SAR obligation
- MLR 2017       Money Laundering Regulations — CDD, EDD, PEP obligations
- JMLSG 2022     Joint Money Laundering Steering Group guidance
- FCA SYSC 6.3   Systems and controls — financial crime
- FATF Rec 10    Customer Due Diligence
- FATF Rec 12    Politically Exposed Persons
- FATF Rec 16    Wire transfer rules
- FATF Rec 19    Higher-risk countries
- FATF Rec 24    Beneficial ownership (legal persons)
- PSR 2017       Payment Services Regulations

Rule catalogue (R01–R10)
-------------------------
R01  CTR_PROXIMITY        £9,500–£9,999 structuring window
R02  ROUND_AMOUNT         Divisible by £1,000
R03  HIGH_RISK_COUNTRY    Counterparty in FATF high-risk jurisdiction
R04  PEP_HIGH_VALUE       PEP account, amount > £25,000
R05  CROSS_BORDER_LARGE   SWIFT/wire cross-border > £50,000
R06  UNKNOWN_PURPOSE      Transaction purpose recorded as 'Unknown'
R07  WEEKEND_WIRE         CHAPS/SWIFT/Wire on Saturday or Sunday
R08  HIGH_RISK_ACCOUNT    Account risk rating = 'High'
R09  OPAQUE_ENTITY_LARGE  SPV or Trust, amount > £100,000
R10  FX_LARGE             Non-GBP transaction > £25,000
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CTR_LOWER             =  9_500.0
CTR_UPPER             = 10_000.0
PEP_THRESHOLD         = 25_000.0
CROSS_BORDER_THRESHOLD= 50_000.0
OPAQUE_THRESHOLD      = 100_000.0
FX_THRESHOLD          =  25_000.0

HIGH_RISK_COUNTRIES = {
    "BVI","Cayman Islands","Cyprus","Malta","Panama",
    "Isle of Man","Seychelles","Vanuatu","Belize","Samoa",
}
WIRE_TYPES  = {"SWIFT","Wire Transfer","CHAPS"}
WEEKEND_DAYS= {"Saturday","Sunday"}
OPAQUE_ENTITIES = {"SPV","Trust"}


# ── Rule functions ────────────────────────────────────────────────────────────

def _r01(df): return df["amount_gbp"].between(CTR_LOWER, CTR_UPPER - 0.01).astype(int)
def _r02(df): return (df["amount_gbp"] % 1000 == 0).astype(int)
def _r03(df): return df["counterparty_country"].isin(HIGH_RISK_COUNTRIES).astype(int)
def _r04(df): return ((df["is_pep"]==1) & (df["amount_gbp"]>PEP_THRESHOLD)).astype(int)
def _r05(df):
    cross = df["account_country"] != df["counterparty_country"]
    wire  = df["transaction_type"].isin(WIRE_TYPES)
    large = df["amount_gbp"] > CROSS_BORDER_THRESHOLD
    return (cross & wire & large).astype(int)
def _r06(df): return (df["purpose"].str.strip().str.lower()=="unknown").astype(int)
def _r07(df): return (df["day_of_week"].isin(WEEKEND_DAYS) & df["transaction_type"].isin(WIRE_TYPES)).astype(int)
def _r08(df): return (df["risk_rating"]=="High").astype(int)
def _r09(df): return (df["entity_type"].isin(OPAQUE_ENTITIES) & (df["amount_gbp"]>OPAQUE_THRESHOLD)).astype(int)
def _r10(df): return ((df["currency"]!="GBP") & (df["amount_gbp"]>FX_THRESHOLD)).astype(int)


RULE_REGISTRY = {
    "R01_CTR_PROXIMITY":      _r01,
    "R02_ROUND_AMOUNT":       _r02,
    "R03_HIGH_RISK_COUNTRY":  _r03,
    "R04_PEP_HIGH_VALUE":     _r04,
    "R05_CROSS_BORDER_LARGE": _r05,
    "R06_UNKNOWN_PURPOSE":    _r06,
    "R07_WEEKEND_WIRE":       _r07,
    "R08_HIGH_RISK_ACCOUNT":  _r08,
    "R09_OPAQUE_ENTITY":      _r09,
    "R10_FX_LARGE":           _r10,
}

RULE_WEIGHTS = {
    "R01_CTR_PROXIMITY":      22,
    "R02_ROUND_AMOUNT":       10,
    "R03_HIGH_RISK_COUNTRY":  20,
    "R04_PEP_HIGH_VALUE":     18,
    "R05_CROSS_BORDER_LARGE": 12,
    "R06_UNKNOWN_PURPOSE":     5,
    "R07_WEEKEND_WIRE":        4,
    "R08_HIGH_RISK_ACCOUNT":   5,
    "R09_OPAQUE_ENTITY":      14,
    "R10_FX_LARGE":            4,
}

RULE_META = {
    "R01_CTR_PROXIMITY":      ("POCA 2002 s.330 | JMLSG 6.7",         "Structuring"),
    "R02_ROUND_AMOUNT":       ("JMLSG Part I 6.7",                     "Unusual Pattern"),
    "R03_HIGH_RISK_COUNTRY":  ("FATF Rec.19 | FCA FC Guide 3.2",       "Jurisdiction"),
    "R04_PEP_HIGH_VALUE":     ("FATF Rec.12 | MLR 2017 Reg.35",        "PEP"),
    "R05_CROSS_BORDER_LARGE": ("FATF Rec.16 | PSR 2017 | SWIFT gpi",   "Cross-Border"),
    "R06_UNKNOWN_PURPOSE":    ("JMLSG Part I 5.3",                     "Documentation"),
    "R07_WEEKEND_WIRE":       ("JMLSG Part I 6.7",                     "Timing"),
    "R08_HIGH_RISK_ACCOUNT":  ("FCA SYSC 6.3 | MLR 2017 Reg.33",       "Account Risk"),
    "R09_OPAQUE_ENTITY":      ("FATF Rec.24 | Companies Act 2006",      "Beneficial Owner"),
    "R10_FX_LARGE":           ("JMLSG Part I 6.7",                     "FX Risk"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════════════════

def apply_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all ten compliance rules.

    Adds columns:
      R01_CTR_PROXIMITY … R10_FX_LARGE  (binary, 0/1)
      rule_flag_count   total rules triggered
      rule_score        weighted composite 0–100
      rule_risk_tier    Low / Medium / High / Critical
      triggered_rules   pipe-separated rule IDs
    """
    df   = df.copy()
    total_w = sum(RULE_WEIGHTS.values())

    for rule_id, fn in RULE_REGISTRY.items():
        try:
            df[rule_id] = fn(df)
        except Exception as e:
            logger.warning("Rule %s failed: %s", rule_id, e)
            df[rule_id] = 0

    df["rule_flag_count"] = sum(df[r] for r in RULE_REGISTRY)
    df["rule_score"]      = sum(
        df[r] * (RULE_WEIGHTS[r] / total_w * 100)
        for r in RULE_REGISTRY
    ).round(1)

    df["rule_risk_tier"] = pd.cut(
        df["rule_score"],
        bins=[-1, 15, 35, 55, 101],
        labels=["Low", "Medium", "High", "Critical"],
    )

    df["triggered_rules"] = df.apply(
        lambda row: "|".join(r for r in RULE_REGISTRY if row.get(r, 0) == 1) or "None",
        axis=1,
    )

    logger.info(
        "Rules applied — Critical:%d  High:%d  Medium:%d  Low:%d",
        (df["rule_risk_tier"]=="Critical").sum(), (df["rule_risk_tier"]=="High").sum(),
        (df["rule_risk_tier"]=="Medium").sum(),   (df["rule_risk_tier"]=="Low").sum(),
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# SAR Candidate Extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_sar_candidates(df: pd.DataFrame, min_score: float = 40.0) -> pd.DataFrame:
    """
    Extract SAR candidates per POCA 2002 s.330.

    Criteria (any one sufficient):
      A. rule_score ≥ min_score
      B. Structuring (R01) AND high-risk jurisdiction (R03)
      C. PEP (R04) with ≥ 2 flags total
    """
    mask = (
        (df["rule_score"] >= min_score)
        | ((df.get("R01_CTR_PROXIMITY", 0)==1) & (df.get("R03_HIGH_RISK_COUNTRY", 0)==1))
        | ((df.get("R04_PEP_HIGH_VALUE", 0)==1) & (df["rule_flag_count"] >= 2))
    )
    result = df[mask].sort_values("rule_score", ascending=False)
    logger.info("%d SAR candidates extracted", len(result))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Effectiveness Report
# ══════════════════════════════════════════════════════════════════════════════

def rule_effectiveness_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute precision, recall, and fraud capture rate for each rule
    against the ground-truth is_fraud label.
    """
    total_fraud = df["is_fraud"].sum()
    rows = []
    for rule_id in RULE_REGISTRY:
        if rule_id not in df.columns:
            continue
        flagged = df[df[rule_id] == 1]
        tp      = (flagged["is_fraud"] == 1).sum()
        fp      = (flagged["is_fraud"] == 0).sum()
        n       = len(flagged)
        prec    = tp / n          if n > 0           else 0
        recall  = tp / total_fraud if total_fraud > 0 else 0
        ref, cat= RULE_META.get(rule_id, ("",""))
        rows.append({
            "rule":           rule_id,
            "category":       cat,
            "total_flagged":  n,
            "true_positives": int(tp),
            "false_positives":int(fp),
            "precision":      round(prec, 4),
            "recall":         round(recall, 4),
            "regulatory_ref": ref,
        })
    return pd.DataFrame(rows).sort_values("precision", ascending=False).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════════════════════

def plot_rule_heatmap(df: pd.DataFrame, save_path: str | None = None):
    """Heatmap: rule trigger rate (%) by sector."""
    rule_cols = [r for r in RULE_REGISTRY if r in df.columns]
    sector_rules = df.groupby("sector")[rule_cols].mean().mul(100)

    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(sector_rules, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Trigger Rate (%)"})
    ax.set_title("Compliance Rule Trigger Rate (%) by Sector", fontsize=14)
    ax.set_xlabel("Rule"); ax.set_ylabel("Sector")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_rule_precision(report: pd.DataFrame, save_path: str | None = None):
    """Horizontal bar chart: rule precision with colour-coded performance."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#27AE60" if p >= 0.60 else "#F39C12" if p >= 0.30 else "#E74C3C"
              for p in report["precision"]]
    ax.barh(report["rule"], report["precision"] * 100,
            color=colors, edgecolor="white")
    ax.axvline(50, color="grey", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Precision (%)")
    ax.set_title("Rule Engine: Precision vs Ground Truth", fontsize=14)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_risk_tier_breakdown(df: pd.DataFrame, save_path: str | None = None):
    """Stacked bar: rule risk tier by entity type."""
    if "rule_risk_tier" not in df.columns:
        return
    ct = pd.crosstab(df["entity_type"], df["rule_risk_tier"])
    tier_order = [t for t in ["Low","Medium","High","Critical"] if t in ct.columns]
    ct = ct[tier_order]
    colors = ["#27AE60","#F39C12","#E67E22","#E74C3C"][:len(tier_order)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("Rule Risk Tier by Entity Type", fontsize=14)
    ax.set_ylabel("Transaction Count"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Risk Tier", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
