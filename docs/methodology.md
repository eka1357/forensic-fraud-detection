# Technical Methodology

## System Architecture

```
Synthetic / Real Transaction Data
            │
            ▼
  src/data_loader.py
  ├── Schema validation
  ├── Dtype normalisation
  └── Derived columns (log_amount, is_weekend, is_cross_border, etc.)
            │
            ▼
  src/statistical_detection.py
  ├── Benford's Law Analysis (Nigrini MAD threshold)
  ├── Z-Score detection (parametric, |z| > 3)
  ├── IQR / Tukey fences (k=1.5 inner, k=3.0 outer)
  ├── Isolation Forest (unsupervised, contamination=0.15)
  ├── Peer Group Benchmarking (sector-level z-score)
  └── Composite Statistical Score (0–100, weighted)
            │
            ▼
  src/rule_based_detection.py
  ├── R01 CTR Proximity    (POCA 2002 s.330)
  ├── R02 Round Amount     (JMLSG 6.7)
  ├── R03 High-Risk Country (FATF Rec.19)
  ├── R04 PEP High-Value   (FATF Rec.12 / MLR 2017)
  ├── R05 Cross-Border     (FATF Rec.16 / PSR 2017)
  ├── R06 Unknown Purpose  (JMLSG 5.3)
  ├── R07 Weekend Wire     (JMLSG 6.7)
  ├── R08 High-Risk Account (FCA SYSC 6.3)
  ├── R09 Opaque Entity    (FATF Rec.24)
  ├── R10 FX Large         (JMLSG 6.7)
  ├── Rule Score (weighted composite)
  └── SAR candidate extraction
            │
            ▼
  src/ml_classifier.py
  ├── Feature matrix (raw + statistical + rule flags)
  ├── sklearn ColumnTransformer (OHE + scaling)
  ├── Random Forest (class_weight='balanced')
  ├── XGBoost (scale_pos_weight=5)
  └── Evaluation: ROC-AUC, PR-AUC, F1, recall
            │
            ▼
  Excel Workbook (7 sheets)     Tableau Dashboard (3 views)
  ├── Executive Summary          ├── Anomaly Overview
  ├── Transaction Detail         ├── Jurisdiction & Entity Risk
  ├── Rule Engine Results        └── Temporal Trends
  ├── Benford Analysis
  ├── Sector Risk Profile
  ├── SAR Priority Queue
  └── Tableau Export
```

---

## Benford's Law

**Theoretical basis**: Benford (1938) showed that in naturally occurring numeric data, the leading significant digit d occurs with probability P(d) = log₁₀(1 + 1/d).

**Implementation**:
```
P(1) = 30.1%, P(2) = 17.6%, P(3) = 12.5%, P(4) = 9.7%,
P(5) = 7.9%,  P(6) = 6.7%,  P(7) = 5.8%,  P(8) = 5.1%, P(9) = 4.6%
```

**Nigrini (2012) MAD thresholds**:
| MAD Range | Conformity |
|-----------|-----------|
| < 0.006 | Close Conformity |
| 0.006–0.012 | Acceptable Conformity |
| 0.012–0.015 | Marginally Acceptable |
| ≥ 0.015 | Non-Conformity — Investigate |

**Test statistic**: Chi-squared goodness-of-fit with 8 degrees of freedom.

**Application**: Benford analysis applied to `amount_gbp`. Digits 8 and 9 over-representation detected with MAD = 0.0193 (non-conformity zone), consistent with structuring below the £10,000 CTR threshold.

---

## Z-Score Detection

**Formula**: z = (x − μ) / σ where μ and σ are the sample mean and standard deviation.

**Threshold**: |z| > 3.0 (parametric, assuming approximate normality of log-amounts).

**Limitation**: Assumes unimodal distribution. Poor performance when fraud amounts cluster near the mean (e.g., structuring). Supplemented by IQR method.

---

## IQR Tukey Fences

**Inner fence**: [Q1 − 1.5·IQR, Q3 + 1.5·IQR]  
**Outer fence**: [Q1 − 3.0·IQR, Q3 + 3.0·IQR]

**Advantage**: Non-parametric — does not assume normality. Robust to skewed distributions typical of financial transaction amounts.

---

## Isolation Forest

**Algorithm**: Liu et al. (2008) — builds an ensemble of isolation trees. Anomalies are isolated in fewer splits than normal observations. Score = inverse of average path length.

**Parameters**:
- `n_estimators = 300`
- `contamination = 0.15` (expected fraud rate)
- Features used: `amount_gbp`, `log_amount`, `is_pep`, `is_cross_border`, `is_high_risk_cpty`, `is_round_amount`, `is_near_ctr`

**Output**: `if_score` (higher = more anomalous) and binary `if_flag`.

---

## Rule Engine Weighting

Rule weights are proportional to regulatory severity and financial crime risk:

| Rule | Weight | Rationale |
|------|--------|-----------|
| R01 CTR Proximity | 22 | Direct POCA s.330 trigger |
| R03 High-Risk Country | 20 | FATF high-risk → EDD mandatory |
| R04 PEP High-Value | 18 | Criminal/political exposure |
| R09 Opaque Entity | 14 | Beneficial ownership opacity |
| R05 Cross-Border | 12 | Wire transfer monitoring |
| R02 Round Amount | 10 | Anomaly indicator |
| R06 Unknown Purpose | 5 | Documentation gap |
| R08 High-Risk Account | 5 | Pre-existing risk flag |
| R07 Weekend Wire | 4 | Timing anomaly |
| R10 FX Large | 4 | FX conversion exposure |

---

## ML Feature Matrix

**Total features after encoding**: ~65–80 (varies by OHE cardinality)

**Feature groups**:
| Group | Count | Examples |
|-------|-------|---------|
| Raw numeric | 8 | amount_gbp, log_amount, is_pep |
| Statistical flags | 5 | if_score, amount_zscore, peer_zscore, rule_score |
| Rule flags | 10 | R01–R10 binary columns |
| OHE categorical | ~30 | sector_*, entity_type_*, transaction_type_* |
| Binary engineered | 6 | is_weekend, is_cross_border, is_round_amount, etc. |

**Class imbalance handling**: XGBoost uses `scale_pos_weight = 5` (approximately 1/fraud_rate). Random Forest uses `class_weight='balanced'`. No SMOTE applied — preserves original distribution in evaluation.

**Validation**: 80/20 stratified train/test split. No data leakage — preprocessing fitted on training folds only.

---

## Regulatory References

| Regulation | Application in Project |
|-----------|----------------------|
| POCA 2002 s.330 | SAR filing trigger; structuring detection (R01) |
| POCA 2002 s.333A | Tipping off offence — SAR confidentiality |
| MLR 2017 Reg.33 | EDD for high-risk third countries (R03) |
| MLR 2017 Reg.35 | PEP EDD and senior management approval (R04) |
| JMLSG Part I 5.3 | Transaction purpose documentation (R06) |
| JMLSG Part I 6.7 | Unusual pattern indicators (R02, R07, R10) |
| FCA SYSC 6.3 | Systems and controls — financial crime |
| FCA FCG Chapter 3 | Fraud risk indicators |
| FATF Rec.10 | Customer Due Diligence |
| FATF Rec.12 | Politically Exposed Persons |
| FATF Rec.16 | Wire transfer transparency |
| FATF Rec.19 | Higher-risk countries |
| FATF Rec.24 | Beneficial ownership — legal persons |
| PSR 2017 | Payment Services Regulations |
