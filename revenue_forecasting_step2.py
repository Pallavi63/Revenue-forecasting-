# ============================================================
# Revenue Forecasting Using ARIMA
# Step 2: ARIMA Model — Build, Evaluate & Forecast
# Author: Burugu Pallavi
# ============================================================

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

# ── COLORS ───────────────────────────────────────────────────
CYAN  = '#00B4D8'
AMBER = '#F4A825'
GREEN = '#2ECC71'
RED   = '#E74C3C'
WHITE = '#FFFFFF'
BG    = '#1B2631'
CARD  = '#212F3D'
GRID  = '#3D5166'

# ── 1. LOAD & FILTER CLEAN MONTHS ────────────────────────────
monthly = pd.read_csv(
    r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\monthly_revenue.csv",
    parse_dates=['date'])
monthly = monthly.sort_values('date').reset_index(drop=True)

# Keep only Aug 2015 — Jun 2016 (11 complete months)
# Exclude Jan-Jul 2015 (incomplete) and Jul 2016 (incomplete)
clean = monthly[
    (monthly['date'] >= '2015-08-01') &
    (monthly['date'] <= '2016-06-01')
].reset_index(drop=True)

print("=" * 55)
print("STEP 1: CLEAN TIME SERIES (11 months)")
print("=" * 55)
print(clean[['date', 'total_revenue']].to_string(index=False))
print(f"\nMonths used for modelling: {len(clean)}")


# ── 2. STATIONARITY TEST (ADF Test) ──────────────────────────
# ARIMA requires the data to be stationary
# ADF test checks if it is — p-value < 0.05 means stationary
print("\n" + "=" * 55)
print("STEP 2: STATIONARITY TEST (ADF Test)")
print("=" * 55)

result = adfuller(clean['total_revenue'])
print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.4f}")
if result[1] < 0.05:
    print("Result        : STATIONARY — data is ready for ARIMA")
else:
    print("Result        : NON-STATIONARY — ARIMA will difference the data (d=1)")


# ── 3. AUTO ARIMA — FIND BEST (p,d,q) ────────────────────────
# Instead of manually testing hundreds of combinations,
# auto_arima finds the best parameters automatically
print("\n" + "=" * 55)
print("STEP 3: AUTO ARIMA — Finding best (p,d,q)")
print("=" * 55)

auto_model = auto_arima(
    clean['total_revenue'],
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None,           # auto-determine differencing
    seasonal=False,   # no seasonality with 11 months
    stepwise=True,
    information_criterion='aic',
    error_action='ignore',
    suppress_warnings=True
)

print(f"Best ARIMA order : {auto_model.order}")
print(f"AIC Score        : {auto_model.aic():.2f}")
print("\nModel Summary:")
print(auto_model.summary())


# ── 4. FIT FINAL ARIMA MODEL ──────────────────────────────────
p, d, q = auto_model.order
print("\n" + "=" * 55)
print(f"STEP 4: FITTING ARIMA({p},{d},{q})")
print("=" * 55)

model = ARIMA(clean['total_revenue'], order=(p, d, q))
fitted = model.fit()
print(fitted.summary())


# ── 5. FORECAST NEXT 3 MONTHS ────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: FORECASTING NEXT 3 MONTHS")
print("=" * 55)

forecast_steps = 3
forecast_result = fitted.get_forecast(steps=forecast_steps)
forecast_mean   = forecast_result.predicted_mean
forecast_ci     = forecast_result.conf_int(alpha=0.2)  # 80% confidence

# Generate future dates — Aug, Sep, Oct 2016
last_date    = clean['date'].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=forecast_steps,
    freq='MS')

forecast_df = pd.DataFrame({
    'date'         : future_dates,
    'forecast'     : forecast_mean.values,
    'lower_bound'  : forecast_ci.iloc[:, 0].values,
    'upper_bound'  : forecast_ci.iloc[:, 1].values
})

print("\nForecast Results:")
print("-" * 55)
for _, row in forecast_df.iterrows():
    print(f"  {row['date'].strftime('%B %Y')} → "
          f"${row['forecast']:>12,.0f}  "
          f"(Range: ${row['lower_bound']:>10,.0f} — ${row['upper_bound']:>10,.0f})")

print(f"\n  Total 3-month forecast : ${forecast_df['forecast'].sum():>12,.0f}")
print(f"  Best case (upper)      : ${forecast_df['upper_bound'].sum():>12,.0f}")
print(f"  Worst case (lower)     : ${forecast_df['lower_bound'].sum():>12,.0f}")


# ── 6. MODEL ACCURACY — MAPE ──────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: MODEL ACCURACY (MAPE)")
print("=" * 55)

fitted_values = fitted.fittedvalues
actual_values = clean['total_revenue'].iloc[d:]  # skip differenced rows

mape = np.mean(np.abs(
    (actual_values.values - fitted_values.values) /
    actual_values.values)) * 100

print(f"MAPE (Mean Absolute Percentage Error) : {mape:.2f}%")
print(f"Model Accuracy                        : {100 - mape:.2f}%")
if mape < 10:
    print("Rating : Excellent — under 10% error")
elif mape < 20:
    print("Rating : Good — under 20% error")
else:
    print("Rating : Acceptable for limited dataset size")


# ── 7. VISUALISE FORECAST ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor(BG)
fig.suptitle('Revenue Forecast — ARIMA Model',
             fontsize=16, color=WHITE, fontweight='bold')

def style_ax(ax, title, ylabel):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=WHITE, fontsize=12,
                 pad=10, fontweight='bold')
    ax.set_ylabel(ylabel, color=WHITE, fontsize=10)
    ax.set_xlabel('Month', color=WHITE, fontsize=10)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=45, ha='right', color=WHITE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.5)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x/1e6:.2f}M'))
    ax.tick_params(axis='y', colors=WHITE)

# Chart 1 — Actual vs Forecast
ax1 = axes[0]
style_ax(ax1, 'Actual Revenue vs ARIMA Forecast', 'Revenue')

# Historical actual line
ax1.plot(clean['date'], clean['total_revenue'],
         color=CYAN, linewidth=2.5, marker='o',
         markersize=6, label='Actual Revenue', zorder=5)
ax1.fill_between(clean['date'], clean['total_revenue'],
                 alpha=0.15, color=CYAN)

# Forecast line — connects from last actual point
connect_dates   = [clean['date'].iloc[-1]] + list(future_dates)
connect_revenue = [clean['total_revenue'].iloc[-1]] + list(forecast_mean.values)
ax1.plot(connect_dates, connect_revenue,
         color=AMBER, linewidth=2.5, marker='o', markersize=6,
         linestyle='--', label='Forecasted Revenue', zorder=5)

# Confidence interval band
ci_dates_full = [clean['date'].iloc[-1]] + list(future_dates)
ci_lower = [clean['total_revenue'].iloc[-1]] + list(forecast_ci.iloc[:, 0].values)
ci_upper = [clean['total_revenue'].iloc[-1]] + list(forecast_ci.iloc[:, 1].values)
ax1.fill_between(ci_dates_full, ci_lower, ci_upper,
                 alpha=0.2, color=AMBER, label='80% Confidence Interval')

# Vertical divider line
ax1.axvline(x=clean['date'].iloc[-1], color=WHITE,
            linewidth=1, linestyle=':', alpha=0.6)
ax1.text(clean['date'].iloc[-1], ax1.get_ylim()[1] * 0.95,
         ' Forecast →', color=WHITE, fontsize=8, alpha=0.8)

# Value labels on forecast points
for date, val in zip(future_dates, forecast_mean.values):
    ax1.annotate(f'${val/1e6:.2f}M',
                 xy=(date, val), xytext=(0, 12),
                 textcoords='offset points',
                 ha='center', fontsize=8,
                 color=AMBER, fontweight='bold')

leg = ax1.legend(facecolor=CARD, fontsize=8, framealpha=0.9)
for text in leg.get_texts():
    text.set_color(WHITE)
leg.get_frame().set_edgecolor(GRID)

# Chart 2 — Forecast Table Visual
ax2 = axes[1]
ax2.set_facecolor(CARD)
ax2.set_title('3-Month Forecast Summary',
              color=WHITE, fontsize=12, pad=10, fontweight='bold')
ax2.axis('off')

table_data = []
for _, row in forecast_df.iterrows():
    table_data.append([
        row['date'].strftime('%B %Y'),
        f"${row['forecast']:,.0f}",
        f"${row['lower_bound']:,.0f}",
        f"${row['upper_bound']:,.0f}"
    ])

table_data.append([
    'TOTAL',
    f"${forecast_df['forecast'].sum():,.0f}",
    f"${forecast_df['lower_bound'].sum():,.0f}",
    f"${forecast_df['upper_bound'].sum():,.0f}"
])

col_labels = ['Month', 'Forecast', 'Worst Case', 'Best Case']
table = ax2.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.4, 2.8)

# Style the table
for (row, col), cell in table.get_celld().items():
    cell.set_facecolor(CARD if row > 0 else '#00B4D8')
    cell.set_text_props(color=WHITE if row > 0 else BG,
                        fontweight='bold' if row == 0 else 'normal')
    cell.set_edgecolor(GRID)
    if row == len(table_data):  # Total row
        cell.set_facecolor('#212F3D')
        cell.set_text_props(color=AMBER, fontweight='bold')

# Model accuracy annotation
ax2.text(0.5, 0.08,
         f"Model: ARIMA{auto_model.order}   |   "
         f"Accuracy: {100-mape:.1f}%   |   MAPE: {mape:.1f}%",
         transform=ax2.transAxes,
         ha='center', fontsize=10, color=WHITE,
         bbox=dict(boxstyle='round', facecolor=GRID,
                   edgecolor=CYAN, alpha=0.8))

plt.tight_layout(pad=3.0)

output_path = r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\step2_arima_forecast.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print(f"\nChart saved to: {output_path}")


# ── 8. EXPORT FORECAST TO CSV ────────────────────────────────
forecast_df['model']    = f'ARIMA{auto_model.order}'
forecast_df['mape']     = round(mape, 2)
forecast_df['accuracy'] = round(100 - mape, 2)

export_path = r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\revenue_forecast.csv"
forecast_df.to_csv(export_path, index=False)
print(f"Forecast exported to: {export_path}")
print("\nStep 2 complete. Ready for category forecasting.")
