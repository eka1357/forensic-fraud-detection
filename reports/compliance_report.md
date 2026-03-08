# Forensic Fraud Detection — Compliance Report

**Classification**: RESTRICTED — Internal Compliance Use Only  
**Analysis Period**: 01 January 2023 – 31 December 2023  
**Dataset**: 2,500 wholesale banking transactions  
**Prepared by**: Financial Analytics Team  
**Date**: 2024  

---

## 1. Executive Summary

This report presents the findings of a forensic analysis of 2,500 wholesale lending transactions using a multi-layered detection framework comprising statistical anomaly detection, a rule-based compliance engine (10 rules), and supervised machine learning classification.

**401 transactions (16.0%)** were confirmed as fraudulent across six distinct fraud patterns. The analysis identified systematic structuring behaviour, Benford's Law non-conformity, and significant exposure to high-risk jurisdictions — each requiring action under UK financial crime regulations.

### Key Metrics

| Metric | Value |
|--------|-------|
| Total transactions analysed | 2,500 |
| Total value analysed | ~£87M |
| Confirmed fraud transactions | 401 (16.0%) |
| SAR candidates | ~240 |
| Critical risk tier | ~120 |
| Highest-risk sector | Import/Export |
| Benford MAD | 0.0193 (NON-CONFORMITY) |
| ML classifier ROC-AUC | ~0.95 (XGBoost) |

---

## 2. Fraud Pattern Analysis

### 2.1 Structuring (POCA 2002 s.330)

**110 transactions (27.4% of all fraud)** were identified in the £9,500–£9,999.99 range — the structuring window below the UK's £10,000 CTR reporting threshold.

All 110 structuring transactions were correctly identified by Rule R01 (precision: 100%). The pattern is systematic and deliberate — random variation would not produce this clustering. The distribution shows a sharp cliff at £10,000 consistent with threshold avoidance.

**Regulatory obligation**: POCA 2002 s.330 requires disclosure where a person knows or suspects, or has reasonable grounds for knowing or suspecting, that money laundering is occurring. SAR filing with the National Crime Agency (NCA) is required. Tipping off the subject is a criminal offence under s.333A.

**Recommended action**: File SAR with NCA within 5 working days for all 110 transactions. Freeze account(s) pending NCA consent where applicable.

### 2.2 Round Number Anomalies (JMLSG Part I 6.7)

**107 transactions (26.7% of all fraud)** involved exact round-number amounts (multiples of £1,000). While round amounts occur legitimately, the concentration in this dataset is statistically anomalous.

Rule R02 has a precision of 33% — the remainder are false positives from legitimate round-number invoices. When combined with other risk factors (high-risk country, PEP, high-risk sector), the precision increases substantially.

**Recommended action**: Do not file SARs for round numbers in isolation. Cross-reference with R03, R04, and peer group anomaly scores. Prioritise round-amount transactions from SPV/Trust entities and Import/Export sector.

### 2.3 High-Risk Jurisdiction Exposure (FATF Recommendation 19)

**74 transactions (18.4% of all fraud)** involved counterparties in FATF-monitored or high-risk jurisdictions (BVI, Cayman Islands, Cyprus, Malta, Panama).

Rule R03 achieved 100% precision — every flagged transaction in this category was confirmed fraud. This is the strongest rule by precision.

**Regulatory obligation**: MLR 2017 Regulation 33 requires Enhanced Due Diligence (EDD) for business relationships and transactions involving high-risk third countries. Senior management approval required for establishing/continuing such relationships.

**Recommended action**: Obtain source of funds documentation for all 74 counterparties. Verify beneficial ownership through Companies House / international equivalents. File SARs immediately for all transactions lacking satisfactory EDD.

### 2.4 Benford's Law Non-Conformity

The full transaction dataset shows a Mean Absolute Deviation (MAD) of **0.0193**, which exceeds Nigrini's (2012) NON-CONFORMITY threshold of 0.015. This is statistically significant (χ² = 89.4, p < 0.0001).

Digit 8 shows Z = +2.8 and digit 9 shows Z = +3.1 — both significantly over-represented relative to the Benford distribution. This is consistent with artificial transaction construction to remain below the £10,000 threshold.

**Forensic note**: Benford's Law analysis is a screening tool, not a direct evidence mechanism. It identifies populations of transactions warranting closer review. Individual Benford violations are not independently actionable without corroborating indicators.

### 2.5 Transaction Velocity Patterns

**50 transactions (12.5% of all fraud)** were identified through high-frequency low-value clustering, consistent with layering behaviour — a common stage in money laundering following initial placement.

**Recommended action**: Conduct network analysis to identify linked accounts in velocity clusters. Consider account-level SAR for accounts showing systematic velocity patterns over a 30-day rolling window.

---

## 3. Sector Risk Profile

| Sector | Fraud Count | Fraud Rate | Risk Tier |
|--------|-------------|------------|-----------|
| Import/Export | 58 | ~23% | CRITICAL |
| Real Estate | 51 | ~20% | CRITICAL |
| Construction | 47 | ~19% | HIGH |
| Finance | 44 | ~18% | HIGH |
| Hospitality | 38 | ~15% | HIGH |
| Professional Services | 35 | ~14% | HIGH |
| Technology | 28 | ~11% | MEDIUM |
| Logistics | 27 | ~11% | MEDIUM |
| Manufacturing | 22 | ~9% | MEDIUM |
| Retail | 18 | ~7% | MEDIUM |

**Import/Export and Real Estate show the highest fraud concentration** — consistent with known UK money laundering typologies involving trade-based money laundering (TBML) and property transactions.

---

## 4. Rule Engine Performance

| Rule | Precision | Recall | Regulatory Basis |
|------|-----------|--------|-----------------|
| R01 CTR Proximity | 100% | 27.4% | POCA 2002 s.330 |
| R03 High-Risk Country | 100% | 18.4% | FATF Rec.19 |
| R09 Opaque Entity | 68% | 12.0% | FATF Rec.24 |
| R02 Round Amount | 33% | 26.7% | JMLSG 6.7 |
| R05 Cross-Border | 27% | 14.7% | FATF Rec.16 |
| R04 PEP High-Value | 7% | 0.2% | MLR 2017 Reg.35 |

Rules R01 and R03 are high-precision instruments suitable for generating automated SAR triggers. Rules R02 and R05 require human review before SAR filing. Rule R04 precision (7%) indicates PEP transactions are largely legitimate in this dataset — the low base rate of PEP transactions (14 total) limits statistical power.

---

## 5. Machine Learning Model Performance

The XGBoost classifier, trained on the combined feature set (raw + statistical + rule-based), achieved:

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.95 |
| Average Precision (PR-AUC) | ~0.88 |
| F1 Score | ~0.83 |
| Recall (fraud capture rate) | ~85% |
| Precision | ~82% |

The model correctly captures approximately **85% of all fraudulent transactions** in holdout testing. The false negative rate (~15%) represents transactions that evaded detection — primarily low-value velocity fraud with insufficient feature density.

**Top 5 predictive features** (by XGBoost gain):
1. `rule_score` — composite rule engine output
2. `if_score` — Isolation Forest anomaly score
3. `is_near_ctr` — structuring window flag
4. `is_high_risk_cpty` — high-risk jurisdiction
5. `log_amount` — log-transformed transaction amount

---

## 6. Compliance Conclusions

### C01 — CRITICAL: File SARs for All Structuring Transactions
110 transactions in the CTR structuring window require immediate SAR filing with the NCA. Deadline: 5 working days from this report date.

### C02 — CRITICAL: Enhanced Due Diligence for High-Risk Jurisdictions
74 transactions lack satisfactory EDD for high-risk jurisdiction counterparties. Deadline: 15 working days. If EDD cannot be completed, terminate relationship and file SAR.

### C03 — HIGH: Benford Non-Conformity Investigation
The forensic digit analysis indicates systematic artificial transaction construction. A targeted investigation of all transactions with leading digit 8 or 9 above £50,000 is recommended (approximately 60 transactions).

### C04 — HIGH: Sector Risk Assessment Update
Import/Export and Real Estate risk ratings should be upgraded to HIGH in the firm's risk appetite framework. New business in these sectors to require Head of Compliance sign-off.

### C05 — MEDIUM: Purpose Documentation Improvement
289 transactions (11.6%) recorded with purpose = "Unknown". JMLSG Part I 5.3 requires firms to understand the purpose of transactions. Implement mandatory purpose field in transaction management system.

### C06 — MEDIUM: PEP Monitoring Enhancement
The low volume (14) of PEP-flagged transactions may indicate under-identification of PEPs. Cross-check client database against HM Treasury Sanctions List and Dow Jones Risk & Compliance.

---

## 7. Regulatory Deliverables

| Deliverable | Status | Deadline |
|-------------|--------|----------|
| SAR — Structuring (110 transactions) | Pending | 5 working days |
| SAR — High-risk jurisdiction (74 transactions) | Pending | Immediate |
| EDD documentation requests (74 counterparties) | Pending | 15 working days |
| Sector risk assessment update | Pending | 30 days |
| PEP database refresh | Pending | 30 days |
| Purpose field system change | Pending | 90 days |

---

## 8. Methodology Note

This analysis uses three complementary detection methods:

1. **Statistical detection** — Benford's Law, Z-score, IQR fences, Isolation Forest, peer group benchmarking
2. **Rule-based detection** — 10 deterministic compliance rules aligned to POCA 2002, MLR 2017, JMLSG 2022, and FATF Recommendations
3. **ML classification** — XGBoost trained on combined feature matrix with SMOTE-adjusted class weighting

Full technical methodology is documented in `docs/methodology.md`.

---

*This report has been prepared for internal compliance use. Distribution is restricted to the Money Laundering Reporting Officer (MLRO), Head of Financial Crime, and Compliance Committee. Unauthorised disclosure of SAR-related information may constitute a criminal offence under POCA 2002 s.333A.*
