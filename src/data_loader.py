"""
src/data_loader.py
------------------
Load, validate, and enrich the transaction dataset.
"""
import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_CSV = os.path.join(DATA_DIR, "sample", "transactions_2500.csv")
TARGET     = "is_fraud"

HIGH_RISK_COUNTRIES = {
    "BVI", "Cayman Islands", "Cyprus", "Malta",
    "Panama", "Isle of Man", "Seychelles", "Vanuatu",
}


def load_data(path: str | None = None) -> pd.DataFrame:
    path = path or SAMPLE_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: '{path}'\nSee data/README.md")

    df = pd.read_csv(path, parse_dates=["date"])
    logger.info("Loaded %d rows × %d columns", *df.shape)

    # Derived columns
    df["log_amount"]         = np.log10(df["amount_gbp"].clip(lower=1))
    df["is_weekend"]         = df["day_of_week"].isin({"Saturday","Sunday"}).astype(int)
    df["is_cross_border"]    = (df["account_country"] != df["counterparty_country"]).astype(int)
    df["is_high_risk_cpty"]  = df["counterparty_country"].isin(HIGH_RISK_COUNTRIES).astype(int)
    df["is_unknown_purpose"] = (df["purpose"].str.lower() == "unknown").astype(int)
    df["is_round_amount"]    = (df["amount_gbp"] % 1000 == 0).astype(int)
    df["is_near_ctr"]        = df["amount_gbp"].between(9500, 9999.99).astype(int)
    df["leading_digit"]      = df["amount_gbp"].apply(_leading_digit)

    return df


def _leading_digit(x: float):
    if x <= 0:
        return None
    s = f"{x:.2f}".lstrip("0").replace(".", "")
    return int(s[0]) if s else None


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype":    df.dtypes,
        "nulls":    df.isnull().sum(),
        "null_pct": df.isnull().mean().mul(100).round(2),
        "unique":   df.nunique(),
    })
