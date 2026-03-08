# Data

## Dataset Overview

| File | Rows | Description |
|------|------|-------------|
| `sample/transactions_2500.csv` | 2,500 | Synthetic transaction dataset (committed) |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| transaction_id | string | Unique transaction identifier (TX######) |
| date | date | Transaction date (2023-01-01 to 2023-12-31) |
| time | string | Transaction time (HH:MM:SS) |
| day_of_week | string | Day name (Monday–Sunday) |
| month | string | Month name |
| quarter | string | Q1–Q4 |
| account_id | string | Originating account ID |
| account_holder | string | Entity name |
| entity_type | string | Corporate / SME / Individual / Trust / SPV / etc. |
| sector | string | Industry sector |
| originating_bank | string | Sending bank |
| account_country | string | Originating account country |
| risk_rating | string | Account risk rating (Low / Medium / High) |
| is_pep | int | 1 if Politically Exposed Person |
| counterparty_id | string | Receiving account ID |
| counterparty_bank | string | Receiving bank |
| counterparty_country | string | Receiving country |
| transaction_type | string | Wire Transfer / BACS / CHAPS / SWIFT / etc. |
| amount_gbp | float | Transaction amount in GBP |
| currency | string | Original currency (GBP / USD / EUR / AED) |
| reference | string | Transaction reference |
| purpose | string | Stated transaction purpose |
| fraud_type | string | Fraud label (None / Structuring / Round Number / etc.) |
| is_fraud | int | **Target variable** — 1 = fraudulent, 0 = legitimate |

## Fraud Patterns Embedded

| Pattern | Count | Description |
|---------|-------|-------------|
| Structuring | 110 | Amounts £9,500–£9,999.99 (below CTR threshold) |
| Round Number | 107 | Exact multiples of £1,000 |
| High-Risk Jurisdiction | 74 | Counterparty in BVI/Cayman/Cyprus/Malta/Panama |
| Benford Anomaly | 59 | Leading digit 8 or 9 (over-represented) |
| Velocity | 50 | High-frequency low-value clustering |
| PEP Involvement | 1 | PEP account high-value transfer |

## Using Real Data

For production use, replace the sample with real transaction exports:

```bash
# Place files in data/raw/ (excluded from Git)
data/raw/
├── transactions.csv    ← export from your transaction management system
└── account_master.csv  ← account details
```

Adjust column mappings in `src/data_loader.py` to match your schema.

## Regenerating the Sample

```bash
python src/data_generator.py --n 2500 --output data/sample/transactions_2500.csv
```
