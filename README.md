# Forensic Fraud Detection

> End-to-end financial transaction fraud detection using statistical anomaly detection, a rule-based compliance engine, and supervised ML classification — aligned to UK wholesale lending, AML, and regulatory reporting standards.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)](https://xgboost.readthedocs.io)

---

## Project Summary

This project applies three complementary detection layers to wholesale banking transactions:

1. **Statistical Detection** — Benford's Law (Nigrini MAD), Z-Score, IQR Tukey fences, Isolation Forest, sector peer benchmarking
2. **Rule-Based Compliance Engine** — 10 deterministic rules mapped to POCA 2002, MLR 2017, JMLSG 2022, and FATF Recommendations
3. **ML Classification** — Random Forest and XGBoost trained on combined feature matrix (~75 features)

Outputs: SAR candidate queue, sector risk profile, Benford conformity report, Excel compliance workbook (7 sheets), and Tableau dashboard guide.

---

## Repository Structure

```
forensic-fraud-detection/
│
├── data/
│   ├── README.md                    ← Schema and data setup instructions
│   └── sample/
│       └── transactions_2500.csv   ← 2,500-row synthetic dataset (committed)
│
├── src/
│   ├── data_loader.py              ← Load, validate, derive columns
│   ├── statistical_detection.py   ← Benford, Z-Score, IQR, Isolation Forest
│   ├── rule_based_detection.py    ← R01–R10 compliance rules + SAR extraction
│   └── ml_classifier.py           ← XGBoost / Random Forest pipelines
│
├── notebooks/
│   ├── 01_eda_statistical_detection.ipynb  ← EDA + Benford + statistical methods
│   ├── 02_rule_based_detection.ipynb       ← Rule engine + SAR candidates
│   ├── 03_ml_classification.ipynb          ← Training, evaluation, SHAP
│   └── 04_compliance_report.ipynb          ← Forensic conclusions + KPIs
│
├── excel/
│   └── Forensic_Fraud_Detection.xlsx       ← 7-sheet professional workbook
│
├── tableau/
│   └── dashboard_guide.md          ← 3-dashboard Tableau build guide
│
├── reports/
│   └── compliance_report.md        ← Structured forensic compliance report
│
├── docs/
│   └── methodology.md              ← Statistical and regulatory methodology
│
├── models/
│   └── .gitkeep                    ← Trained models saved here (git-excluded)
│
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/<your-username>/forensic-fraud-detection.git
cd forensic-fraud-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run notebooks in order

```bash
jupyter notebook
```

| Notebook | What it covers |
|----------|----------------|
| `01_eda_statistical_detection` | Data quality, Benford's Law, Z-Score, IQR, Isolation Forest |
| `02_rule_based_detection` | All 10 compliance rules, SAR extraction, sector heatmap |
| `03_ml_classification` | XGBoost + Random Forest, ROC/PR curves, feature importance |
| `04_compliance_report` | Executive KPIs, conclusions, SAR priority queue |

### 3. Open Excel workbook

`excel/Forensic_Fraud_Detection.xlsx` contains 7 pre-built sheets:
- **Executive Summary** — KPI tiles, findings table
- **Transaction Detail** — Top 150 highest-risk transactions
- **Rule Engine Results** — All 10 rules with precision/recall
- **Benford Analysis** — Digit distribution with chart
- **Sector Risk Profile** — Fraud rate by industry
- **SAR Priority Queue** — Filing deadlines by fraud type
- **Tableau Export** — Clean data for Tableau connection

### 4. Build Tableau Dashboards

Follow `tableau/dashboard_guide.md` to connect the workbook and build 3 dashboards:
1. Anomaly Overview (KPIs, fraud by type, risk tiers, amount distribution)
2. Jurisdiction & Entity Risk (map, sector×entity heatmap, treemap)
3. Temporal & Compliance Trends (monthly volume, rule triggers, running totals)

---

## Compliance Framework Coverage

| Regulation | Rules / Methods Applied |
|-----------|------------------------|
| POCA 2002 s.330 | R01 (structuring detection), SAR candidate extraction |
| MLR 2017 Reg.33/35 | R03 (EDD), R04 (PEP) |
| JMLSG Part I 5.3 & 6.7 | R02, R06, R07, R10 |
| FCA SYSC 6.3 | R08 (high-risk account monitoring) |
| FATF Rec. 10/12/16/19/24 | R03, R04, R05, R09 |
| PSR 2017 | R05 (cross-border wire) |

---

## Detection Results (Sample Data)

| Detection Method | Flagged | Precision | Recall | Notes |
|-----------------|---------|-----------|--------|-------|
| Benford's Law | Diagnostic | — | — | MAD=0.0193 (non-conformity) |
| Z-Score (\|z\|>3) | ~85 | ~38% | ~8% | Gross outliers only |
| IQR Inner Fence | ~180 | ~32% | ~14% | Structural outliers |
| Isolation Forest | ~375 | ~43% | ~40% | Best unsupervised |
| Rule R01 (Structuring) | 110 | 100% | 27% | Highest precision |
| Rule R03 (High-Risk) | 74 | 100% | 18% | Perfect detection |
| Rule Engine (all) | ~600 | ~45% | ~55% | Combined rules |
| XGBoost Classifier | ~340 | ~82% | ~85% | Best overall |

---

## Key Forensic Findings

1. **Systematic structuring**: 110 transactions concentrated in £9,500–£9,999 window — SAR-reportable under POCA 2002 s.330
2. **Benford non-conformity**: MAD = 0.0193 exceeds Nigrini's NON-CONFORMITY threshold; digits 8 and 9 statistically over-represented
3. **High-risk jurisdiction exposure**: 74 transactions to BVI/Cayman/Cyprus/Malta require EDD under MLR 2017
4. **Import/Export sector**: Highest fraud rate (~23%) — consistent with Trade-Based Money Laundering (TBML) typology
5. **XGBoost AUC ~0.95**: Near-production-ready classification performance on combined feature set

---

## Stack

| Category | Tools |
|----------|-------|
| Data generation | Python stdlib (csv, random) |
| Statistical detection | scipy, scikit-learn (IsolationForest), numpy |
| Rule engine | pandas (vectorised operations) |
| ML classification | scikit-learn, XGBoost |
| Visualisation | matplotlib, seaborn |
| Spreadsheet | openpyxl |
| Dashboard | Tableau Desktop / Tableau Public |
| Notebooks | Jupyter |
