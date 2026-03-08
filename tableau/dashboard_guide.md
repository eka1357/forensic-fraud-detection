# Tableau Dashboard — Build Guide

## Data Source

Connect Tableau to the **Tableau Export** sheet in `excel/Forensic_Fraud_Detection.xlsx`.

**Connection steps**:
1. Open Tableau Desktop
2. Connect → Microsoft Excel → select `Forensic_Fraud_Detection.xlsx`
3. Drag "Tableau Export" sheet to canvas
4. Ensure these fields are correctly typed:

| Field | Type | Notes |
|-------|------|-------|
| date | Date | Format: YYYY-MM-DD |
| amount_gbp | Number (Decimal) | |
| is_fraud | Number (Whole) | Will be used as dimension 0/1 |
| rule_score | Number (Decimal) | |
| log_amount | Number (Decimal) | |

---

## Dashboard 1: Anomaly Overview

**Purpose**: Executive-level snapshot of fraud distribution and detection system health.

### Visualisations

#### 1a. KPI Tiles (Text / Number)
| Tile | Calculation |
|------|-------------|
| Total Transactions | COUNT([Transaction Id]) |
| Fraudulent Transactions | COUNTIF([Is Fraud] = 1) |
| Total Value Analysed | SUM([Amount Gbp]) |
| SAR Candidates | COUNTIF([Rule Score] >= 40) |
| Average Rule Score | AVG([Rule Score]) |

#### 1b. Fraud by Type — Horizontal Bar Chart
- Dimension: `fraud_type`
- Measure: COUNT([Transaction Id])
- Filter: `is_fraud = 1`
- Colour: Use diverging palette (Red = high count)
- Sort: Descending by count

#### 1c. Risk Tier Distribution — Pie Chart
- Dimension: `rule_risk_tier`
- Measure: COUNT([Transaction Id])
- Colours: Critical = #E74C3C, High = #F39C12, Medium = #2980B9, Low = #27AE60

#### 1d. Amount Distribution — Histogram
- Bin field: `amount_gbp` (bin size = 2,500)
- Measure: COUNT([Transaction Id])
- Add reference line at 10,000 (CTR threshold)
- Add shaded band 9,500–10,000 (structuring window)
- Filter: Fraud vs Legitimate overlay (dual axis)

#### 1e. Fraud Rate by Sector — Packed Bubble
- Dimension: `sector`
- Size: COUNT([Transaction Id])
- Colour: Calculated field `[Fraud Rate]` = `SUM([Is Fraud])/COUNT([Is Fraud])`

---

## Dashboard 2: Jurisdiction & Entity Risk

**Purpose**: Regulatory risk exposure by geography and entity type.

#### 2a. World Map — Counterparty Country
- Geographic role: Country/Region on `counterparty_country`
- Colour: `SUM([Is Fraud])` — sequential red palette
- Tooltip: Fraud count, fraud rate, total value
- Highlight FATF high-risk jurisdictions with custom shape layer

#### 2b. Heatmap — Sector × Entity Type
- Rows: `sector`
- Columns: `entity_type`
- Colour: `AVG([Rule Score])` — YlOrRd palette
- Label: Rule score value
- This replicates the Python rule heatmap in an interactive format

#### 2c. Treemap — Transaction Value by Sector
- Dimension: `sector`, `entity_type`
- Size: `SUM([Amount Gbp])`
- Colour: `AVG([Is Fraud])` — fraud rate

#### 2d. Scatter Plot — Amount vs Rule Score
- X-axis: `amount_gbp` (log scale)
- Y-axis: `rule_score`
- Colour: `fraud_type`
- Shape: `is_fraud` (fraud = filled circle, legitimate = hollow)
- Filter panel: entity_type, sector, quarter

---

## Dashboard 3: Temporal & Compliance Trends

**Purpose**: Time-series analysis for trend detection and regulatory monitoring.

#### 3a. Monthly Transaction Volume — Dual-Axis Line/Bar
- X-axis: `date` (by month)
- Bar: COUNT([Transaction Id]) — total volume
- Line: `SUM([Is Fraud])/COUNT([Is Fraud])` — fraud rate %
- Dual axis with independent scales

#### 3b. Quarterly Rule Trigger Heatmap
- Rows: Compliance rule (calculated field from `triggered_rules`)
- Columns: `quarter`
- Colour: Trigger rate % per quarter
- Use calculated field: split `triggered_rules` on "|"

#### 3c. Cumulative Fraud Value — Running Total
- X-axis: date (sorted)
- Y-axis: RUNNING_SUM(SUM([Amount Gbp])) where `is_fraud = 1`
- Reference band: Regulatory reporting threshold (£50,000 cumulative)

#### 3d. Structuring Window Time Series
- Filter: `is_near_ctr = 1`
- X-axis: date (by week)
- Y-axis: COUNT([Transaction Id])
- Annotate peaks with transaction count labels

---

## Calculated Fields to Create

```tableau
// Fraud Rate
[Fraud Rate] = SUM([Is Fraud]) / COUNT([Transaction Id])

// Is High Risk Country (Categorical)
[Is High Risk Jurisdiction] =
  IF [Counterparty Country] IN ("BVI", "Cayman Islands", "Cyprus", "Malta", "Panama")
  THEN "High Risk"
  ELSE "Standard"
  END

// Risk Tier Colour (for custom palettes)
[Risk Tier Colour] =
  IF [Rule Risk Tier] = "Critical" THEN "#E74C3C"
  ELSEIF [Rule Risk Tier] = "High" THEN "#F39C12"
  ELSEIF [Rule Risk Tier] = "Medium" THEN "#2980B9"
  ELSE "#27AE60"
  END

// SAR Flag
[SAR Candidate] = IIF([Rule Score] >= 40, "SAR Required", "Monitor")

// CTR Structuring Window
[In Structuring Window] = IIF([Amount Gbp] >= 9500 AND [Amount Gbp] < 10000, "Structuring Risk", "Normal")
```

---

## Filters & Parameters

### Global Filters (apply to all dashboards)
- **Date range**: `date` — default last 12 months
- **Sector**: `sector` — multi-select
- **Entity type**: `entity_type` — multi-select
- **Fraud flag**: `is_fraud` — 0 / 1 / All

### Dashboard Parameters
| Parameter | Type | Values |
|-----------|------|--------|
| Risk Threshold | Float | Min 0, Max 100, default 40 |
| Amount Minimum | Integer | Min 0, Max 500000, default 0 |

---

## Formatting Standards

| Element | Specification |
|---------|--------------|
| Primary font | Tableau Book, 11pt |
| Header font | Tableau Bold, 14pt |
| Background | White (#FFFFFF) |
| Grid lines | Light grey (#F0F0F0) |
| Critical colour | #E74C3C (Red) |
| High colour | #F39C12 (Amber) |
| Medium colour | #2980B9 (Blue) |
| Low/Clean colour | #27AE60 (Green) |
| Fraud highlight | #E74C3C with 80% opacity |
| Legitimate highlight | #1A1A2E with 50% opacity |

---

## Tooltip Templates

### Transaction Tooltips
```
Transaction: <Transaction Id>
Date: <Date>
Sector: <Sector>
Amount: £<Amount Gbp>
Fraud Type: <Fraud Type>
Rule Score: <Rule Score>
Risk Tier: <Rule Risk Tier>
```

### SAR Reference Tooltip
```
SAR Status: <SAR Candidate>
Regulatory Basis: POCA 2002 s.330
Filing Deadline: <based on fraud type>
```

---

## Publishing

1. Save as `.twbx` (packaged workbook) to include Excel data source
2. Publish to Tableau Server / Tableau Public with:
   - Extract refresh: Daily at 06:00
   - Permissions: Internal Compliance Only
   - Content label: RESTRICTED
