# 💊 Pharmacy Demand Forecaster

> **Predicts medicine sales using past demand and seasonal patterns via regression with lag features and rolling dynamics.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest%20%7C%20Scikit--Learn-green.svg)](https://scikit-learn.org/)
[![Interactive Dashboard](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Unit Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)]()

---

## 📖 Overview

The **Pharmacy Demand Forecaster** is an end-to-end machine learning system tailored to predict medication consumption, mitigate stockout risks for life-critical medicines, and calculate resilient safety buffer stock.

Unlike generic retail demand, pharmaceutical sales exhibit distinct operational dynamics:
1. **Epidemiological Seasonality**: Winter respiratory surges (Antibiotics like Amoxicillin, Azithromycin), spring and autumn pollen peaks (Allergy medications like Cetirizine).
2. **Chronic Prescription Refill Waves**: Maintenance therapies (Diabetes: Metformin, Cardiovascular: Atorvastatin, Amlodipine) exhibit monthly refill spikes at month-start (days 1–5) and month-end (days 27–31).
3. **Weekly Clinic Visitation Patterns**: Heightened physician clinic visits on Mondays and Saturdays, with subdued Sunday clinic hours.
4. **Viral / Bacterial Epidemic Surges**: Sudden 2x–3x demand explosions lasting 2 to 4 weeks during disease waves.

---

## 🏗️ Architecture & Feature Engineering Pipeline

To avoid lookahead bias and achieve accurate multi-step forecasting, the system uses strict time-series feature engineering:

### 1. Autoregressive Lag Features
- **Daily Lags**: $t-1, t-2, t-3$ (capturing short-term momentum and inertia)
- **Weekly Cycles**: $t-7, t-14, t-21, t-28$ (capturing strong day-of-week seasonality)

### 2. Rolling Window Aggregates (Zero Data Leakage)
Computed on strictly shifted series ($t-1$ and earlier):
- **Rolling Averages**: 7-day, 14-day, and 28-day moving means
- **Rolling Volatilities**: 7-day, 14-day, and 28-day standard deviations
- **Rolling Bounds**: 7-day minimums and maximums
- **Exponential Moving Averages (EMA)**: 7-day and 14-day EMA for rapid trend adaptation

### 3. Harmonic Cyclical Calendar Encoding
Trigonometric mappings maintain circular continuity (Sunday connects smoothly to Monday, December to January):
- $\sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$ & $\cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$
- $\sin\left(\frac{2\pi \cdot (\text{month}-1)}{12}\right)$ & $\cos\left(\frac{2\pi \cdot (\text{month}-1)}{12}\right)$
- $\sin\left(\frac{2\pi \cdot (\text{day\_of\_year}-1)}{365.25}\right)$ & $\cos\left(\frac{2\pi \cdot (\text{day\_of\_year}-1)}{365.25}\right)$

### 4. Demand Momentum & Acceleration
- **Momentum Ratio**: $\frac{\text{roll\_mean\_7} + \epsilon}{\text{roll\_mean\_28} + \epsilon}$ (early warning indicator for outbreak surges)
- **Day-over-Day Acceleration**: $\text{lag\_1} - \text{lag\_2}$
- **Week-over-Week Delta**: $\text{lag\_1} - \text{lag\_7}$

---

## 📦 Inventory Optimization & Safety Stock

Demand forecasts directly inform pharmaceutical replenishment decisions:

- **Safety Stock ($SS$)**:
  $$SS = Z_{\alpha} \times \sqrt{L} \times \text{RMSE}_{\text{forecast}}$$
  *(where $Z_{\alpha}$ is standard normal score for target service level, e.g. $1.645$ for 95%, $L$ is supplier lead time in days)*

- **Reorder Point ($ROP$)**:
  $$ROP = (\hat{d} \times L) + SS$$
  *(where $\hat{d}$ is forecasted mean daily demand)*

- **Economic Order Quantity ($EOQ$)**:
  $$EOQ = \sqrt{\frac{2 \times D \times S}{H}}$$

- **Stockout Alerting**: Flags inventory health as `CRITICAL_STOCKOUT_RISK`, `REORDER_TRIGGERED`, or `OPTIMAL`.

---

## 📂 Project Directory Structure

```
pharmacy-demand-forecaster/
├── app.py                      # Interactive Streamlit Web Application
├── cli.py                      # Command Line Pipeline & Automation
├── requirements.txt            # Python Dependencies
├── README.md                   # Full Documentation
├── data/
│   ├── generator.py            # Multi-year synthetic pharmacy sales generator
│   ├── loader.py               # Dataset validation and date-imputation loader
│   └── raw_pharmacy_sales.csv  # Generated multi-year dataset (10,960 records)
├── features/
│   ├── __init__.py
│   └── engineer.py             # Lag, rolling, cyclical, & momentum engine
├── models/
│   ├── __init__.py
│   ├── forecaster.py           # Core forecaster (XGBoost, RF, Ridge, GB)
│   ├── evaluator.py            # Metrics (RMSE, MAE, WAPE, MAPE, R², Bias)
│   └── inventory_optimizer.py  # Safety stock & reorder point optimizer
└── tests/
    ├── __init__.py
    ├── test_data_generator.py  # Tests for data generator & loader
    ├── test_features.py        # Tests for lag engineering & leakage prevention
    ├── test_models.py          # Tests for XGBoost training & forecasting
    └── test_inventory.py       # Tests for safety stock calculations
```

---

## 🚀 Quickstart Guide

### 1. Installation

Ensure Python 3.10+ is installed:

```bash
cd pharmacy-demand-forecaster
pip install -r requirements.txt
```

### 2. Run Interactive Web Dashboard (Streamlit)

Launch the UI:

```bash
streamlit run app.py
```

Features included in the dashboard:
- **Demand Explorer & KPIs**: Lifetime sales, 30-day run rates, moving average charts, weekly & monthly seasonal profiles.
- **Future Forecasting**: Multi-step recursive 7/14/30-day projections with 95% confidence bands and CSV export.
- **Holdout Backtest Viewer**: Compare actual sales vs model predictions over a 60-day out-of-time test set.
- **Model Benchmarking**: Side-by-side comparison of XGBoost, Random Forest, Gradient Boosting, and Ridge.
- **Feature Importance Chart**: Relative contributions of lags, rolling averages, and calendar features.
- **Automated Inventory Replenishment**: Dynamic Safety Stock, Reorder Point, and EOQ order recommendations.

---

## 💻 CLI Usage Examples

### 1. Benchmark All Models
Compare all candidate regression models on the same holdout validation split:
```bash
python cli.py --benchmark --medicine MED-001
```

### 2. Train and Evaluate Forecaster
```bash
python cli.py --train --medicine MED-003 --model xgboost
```

### 3. Generate 14-Day Demand Forecast
```bash
python cli.py --forecast --medicine MED-001 --horizon 14
```

### 4. Calculate Safety Stock & Reorder Policy
```bash
python cli.py --inventory --medicine MED-001 --current-stock 80 --service-level 0.95
```

### 5. Combined End-to-End Pipeline
```bash
python cli.py --benchmark --forecast --inventory --medicine MED-001 --horizon 7
```

---

## 🧪 Running Unit Tests

Run the full automated test suite:

```bash
python -m unittest discover -s tests
```

---

## 📊 Supported Medicines in Pre-Loaded Catalog

| ID | Medicine Name | Category | Base Demand | Seasonality Pattern |
|---|---|---|---|---|
| `MED-001` | Amoxicillin 500mg | Antibiotics | 45 | Winter Respiratory Peak |
| `MED-002` | Azithromycin 250mg | Antibiotics | 30 | Winter Respiratory Peak |
| `MED-003` | Paracetamol 650mg | Analgesics & Antipyretics | 120 | Flu & Monsoon Rain Surge |
| `MED-004` | Ibuprofen 400mg | Analgesics & Antipyretics | 65 | Steady Baseline |
| `MED-005` | Cetirizine 10mg | Antihistamines | 50 | Spring & Fall Pollen Surge |
| `MED-006` | Montelukast 10mg | Respiratory & Asthma | 40 | Seasonal Asthma + Refills |
| `MED-007` | Metformin 500mg | Chronic - Diabetes | 95 | Steady + Monthly Refill Cycles |
| `MED-008` | Atorvastatin 20mg | Chronic - Cardiovascular | 75 | Steady + Monthly Refill Cycles |
| `MED-009` | Amlodipine 5mg | Chronic - Cardiovascular | 70 | Steady + Monthly Refill Cycles |
| `MED-010` | Omeprazole 20mg | Gastrointestinal | 55 | Festive & Holiday Spikes |
