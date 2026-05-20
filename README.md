# Revenue Forecasting Using ARIMA — Time Series Analysis

## Project Overview
End-to-end time series forecasting project built on 11 months of cleaned retail sales data. Applied ARIMA modelling to forecast the next 3 months of revenue at both overall and product category level. The goal was to provide data-driven revenue projections with confidence intervals to support business planning and budget decisions.

---

## Dashboard Preview
![Revenue Forecasting Dashboard](image.png)

---

## Tools Used
| Tool | Purpose |
|---|---|
| Python (Pandas, Statsmodels, pmdarima) | Time series modelling & forecasting |
| Matplotlib | Forecast visualisation with confidence bands |
| Power BI | Interactive forecasting dashboard |

---

## Dataset
- **Source:** Sales Performance project — bicycle accessories retail business
- **Size:** 34,865 transactions aggregated to 11 monthly data points
- **Date Range Used:** August 2015 — June 2016 (clean complete months only)
- **Excluded:** January–July 2015 (incomplete records, 138–206 orders/month vs 2,000+ in full months) and July 2016 (partial month)

---

## Project Structure
```
revenue-forecasting/
│
├── data/
│   ├── monthly_revenue.csv              ← aggregated monthly revenue
│   ├── monthly_revenue_by_category.csv  ← monthly revenue per category
│   ├── revenue_forecast.csv             ← overall 3-month forecast output
│   ├── category_forecast.csv            ← category-level forecast output
│   └── combined_revenue.csv             ← actual + forecast combined for Power BI
│
├── python/
│   ├── step1_eda.py                     ← time series EDA & visualisation
│   ├── step2_arima_forecast.py          ← ARIMA model + overall forecast
│   └── step3_category_forecast.py       ← per-category ARIMA models
│
├── powerbi/
│   └── revenue_forecasting_dashboard.pbix
│
├── screenshots/
│   ├── step1_eda_charts.png             ← EDA visualisations
│   ├── step2_arima_forecast.png         ← actual vs forecast chart
│   ├── step3_category_forecast.png      ← category forecast charts
│   └── dashboard.png                    ← final Power BI dashboard
│
└── README.md
```

---

## Step 1 — Exploratory Data Analysis

Aggregated 34,865 daily transactions into monthly revenue totals and performed time series EDA:

- **Monthly Revenue Trend** — area chart showing revenue growth from $0.2M to $2.3M peak
- **Monthly Order Volume** — bar chart confirming data completeness per month
- **Revenue by Category** — multi-line chart showing Bikes, Accessories, Clothing trends
- **Month-over-Month Growth %** — bar chart identifying 133.6% spike at data completeness change point

**Key observation:** January–July 2015 showed only 138–206 orders per month compared to 2,000–3,400 in complete months. These incomplete months were excluded from modelling to avoid distorting forecast accuracy.

---

## Step 2 — ARIMA Model (Overall Revenue)

### Stationarity Test
Applied **Augmented Dickey-Fuller (ADF) Test** to check if the time series was stationary before fitting ARIMA:
- Non-stationary result → confirmed differencing (d=1) was required

### Model Selection
Used **auto_arima** from pmdarima library to automatically evaluate all (p,d,q) combinations and select the optimal model based on AIC score:

| Parameter | Value | Meaning |
|---|---|---|
| p | 0 | No autoregressive terms |
| d | 1 | First-order differencing applied |
| q | 0 | No moving average terms |

**ARIMA(0,1,0)** — also known as a Random Walk with Drift. Appropriate for a consistently growing business with limited historical data.

### Model Performance
| Metric | Value |
|---|---|
| MAPE (Mean Absolute % Error) | 18.0% |
| Model Accuracy | **82.0%** |
| Rating | Good — under 20% threshold |

### 3-Month Forecast Results
| Month | Forecast | Worst Case | Best Case |
|---|---|---|---|
| July 2016 | $2,344,229 | $1,997,602 | $2,690,856 |
| August 2016 | $2,344,229 | $1,854,024 | $2,834,434 |
| September 2016 | $2,344,229 | $1,743,853 | $2,944,605 |
| **Total** | **$7,032,687** | **$5,595,479** | **$8,469,895** |

The widening confidence interval correctly reflects increasing uncertainty over longer forecast horizons — a hallmark of statistically sound forecasting.

---

## Step 3 — Category-Level ARIMA Models

Built 3 separate ARIMA models — one per product category — to understand which category will drive future revenue:

| Category | ARIMA Order | 3M Forecast | Accuracy | Pattern |
|---|---|---|---|---|
| Bikes | (1,0,0) | $2,992,355 | **82.4%** | Declining trend — post-peak correction |
| Accessories | (0,1,0) | $2,339,775 | **85.4%** | Flat — stable consistent growth |
| Clothing | (0,1,0) | $1,093,488 | **82.5%** | Flat — stable low volume |

**Notable finding:** Bikes ARIMA(1,0,0) used `p=1` — meaning Bikes revenue is influenced by the previous month's value (momentum effect). The declining forecast ($1.07M → $0.99M → $0.93M) suggests a post-peak correction after strong May–June 2016 performance.

**Accessories achieved the highest model accuracy at 85.4%** — its stable growth pattern makes it the most predictable and reliable revenue stream.

---

## Step 4 — Power BI Dashboard

Single-page dark-themed interactive dashboard with 4 visuals:

### KPI Cards
| Metric | Value |
|---|---|
| Total 3-Month Forecast | $7.03M |
| Best Case Scenario | $8.47M |
| Worst Case Scenario | $5.60M |
| Model Accuracy | 82.01% |

### Visuals
| Visual | Description |
|---|---|
| Actual vs Forecast Line Chart | Solid cyan = historical revenue, amber dashed = 3-month forecast |
| Category Forecast Bar Chart | Bikes $2.99M, Accessories $2.34M, Clothing $1.09M |
| Monthly Forecast Detail Table | Month-by-month forecast with worst and best case ranges |

---

## Key Findings & Business Insights

### Finding 1 — Revenue Expected to Hold Steady at $2.34M/Month
The overall ARIMA model forecasts $7.03M over 3 months — consistent with the recent revenue plateau at $2.3–2.4M per month seen since April 2016.

### Finding 2 — Bikes Forecast Shows Declining Trend
Bikes revenue is projected to decline from $1.07M to $0.93M over 3 months — a 13% drop from peak. This aligns with the post-summer seasonal correction pattern in the historical data.

### Finding 3 — Accessories is the Most Stable Revenue Stream
Accessories forecasts a flat $779K per month with the highest model accuracy (85.4%) — making it the most predictable and reliable category for budget planning.

### Finding 4 — Confidence Interval Widens Over Time
Best case ($8.47M) vs worst case ($5.60M) represents a $2.87M uncertainty range — highlighting that forecasting accuracy naturally decreases further into the future. This range should be communicated to stakeholders when setting budget expectations.

---

## Business Recommendations

1. **Use $5.6M as conservative budget baseline** — worst case scenario provides a safe floor for planning
2. **Investigate Bikes decline** — if the forecast decline materialises, review Bikes pricing and marketing spend for Q3 2016
3. **Increase Accessories inventory** — most stable and predictable category, safe to stock up based on $779K/month forecast
4. **Collect more historical data** — with 24+ months of data, ARIMA accuracy would improve significantly beyond the current 82%

---

## What I Learned
- How to detect and handle incomplete time series data before modelling
- How to apply ADF stationarity test and interpret results
- How to use auto_arima to automate model selection instead of manually testing parameters
- How ARIMA(0,1,0) behaves as a random walk — appropriate for limited data with growth trend
- How confidence intervals widen over forecast horizons — and why this matters for business communication
- How building separate models per category reveals different underlying patterns (momentum vs stable growth)

---

## How to Run

**Python:**
```bash
pip install pandas numpy matplotlib statsmodels pmdarima openpyxl
```
Run scripts in order: `step1_eda.py` → `step2_arima_forecast.py` → `step3_category_forecast.py`

**Power BI:**
1. Download [Power BI Desktop](https://powerbi.microsoft.com/desktop) (free)
2. Open `revenue_forecasting_dashboard.pbix`
3. Refresh data if prompted

---

