# ============================================================
# Revenue Forecasting Using ARIMA
# Step 3: Category Revenue Forecasting
# Author: Burugu Pallavi
# ============================================================

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.arima.model import ARIMA
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
CAT_COLORS = {'Bikes': CYAN, 'Accessories': AMBER, 'Clothing': GREEN}

# ── 1. LOAD & FILTER CATEGORY DATA ───────────────────────────
category_monthly = pd.read_csv(
    r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\monthly_revenue_by_category.csv",
    parse_dates=['date'])
category_monthly = category_monthly.sort_values(
    ['product_category', 'date']).reset_index(drop=True)

# Keep only Aug 2015 — Jun 2016 (same clean window as overall model)
category_clean = category_monthly[
    (category_monthly['date'] >= '2015-08-01') &
    (category_monthly['date'] <= '2016-06-01')
].reset_index(drop=True)

categories = ['Bikes', 'Accessories', 'Clothing']

print("=" * 55)
print("STEP 1: CATEGORY DATA (11 months each)")
print("=" * 55)
for cat in categories:
    sub = category_clean[category_clean['product_category'] == cat]
    print(f"\n{cat}: {len(sub)} months, "
          f"Avg monthly revenue: ${sub['total_revenue'].mean():,.0f}")


# ── 2. FIT ARIMA PER CATEGORY & FORECAST ─────────────────────
print("\n" + "=" * 55)
print("STEP 2: ARIMA MODELS PER CATEGORY")
print("=" * 55)

forecast_steps = 3
results = {}

for cat in categories:
    sub = category_clean[
        category_clean['product_category'] == cat
    ]['total_revenue'].reset_index(drop=True)

    # Auto ARIMA
    auto_m = auto_arima(sub, start_p=0, max_p=3,
                        start_q=0, max_q=3,
                        d=None, seasonal=False,
                        stepwise=True, suppress_warnings=True,
                        error_action='ignore')

    # Fit model
    model  = ARIMA(sub, order=auto_m.order)
    fitted = model.fit()

    # Forecast
    fc     = fitted.get_forecast(steps=forecast_steps)
    fc_mean= fc.predicted_mean
    fc_ci  = fc.conf_int(alpha=0.2)

    # MAPE
    fv = fitted.fittedvalues
    av = sub
    common = av.index.intersection(fv.index)
    mape = np.mean(np.abs(
        (av.loc[common].values - fv.loc[common].values) /
        av.loc[common].values)) * 100

    results[cat] = {
        'order'       : auto_m.order,
        'fitted'      : fitted,
        'actual'      : sub,
        'fc_mean'     : fc_mean,
        'fc_lower'    : fc_ci.iloc[:, 0],
        'fc_upper'    : fc_ci.iloc[:, 1],
        'mape'        : mape,
        'accuracy'    : 100 - mape
    }

    print(f"\n{cat}:")
    print(f"  ARIMA order : {auto_m.order}")
    print(f"  MAPE        : {mape:.2f}%")
    print(f"  Accuracy    : {100-mape:.2f}%")
    print(f"  Forecast    : {[f'${v:,.0f}' for v in fc_mean.values]}")


# ── 3. BUILD FORECAST DATES ───────────────────────────────────
last_date    = category_clean['date'].max()
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=forecast_steps, freq='MS')

hist_dates = category_clean[
    category_clean['product_category'] == 'Bikes']['date'].values


# ── 4. VISUALISE ALL 3 CATEGORY FORECASTS ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 11))
fig.patch.set_facecolor(BG)
fig.suptitle('Revenue Forecast by Product Category — ARIMA',
             fontsize=16, color=WHITE, fontweight='bold')

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=WHITE, fontsize=12,
                 pad=10, fontweight='bold')
    ax.set_ylabel('Revenue', color=WHITE, fontsize=9)
    ax.set_xlabel('Month',   color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=45, ha='right', color=WHITE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.5)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x/1e6:.2f}M'))
    ax.tick_params(axis='y', colors=WHITE)

# One chart per category
for idx, cat in enumerate(categories):
    ax  = axes[idx // 2][idx % 2]
    res = results[cat]
    col = CAT_COLORS[cat]

    style_ax(ax, f'{cat} — ARIMA{res["order"]} | '
                 f'Accuracy: {res["accuracy"]:.1f}%')

    # Historical
    ax.plot(hist_dates, res['actual'].values,
            color=col, linewidth=2.5, marker='o',
            markersize=5, label='Actual', zorder=5)
    ax.fill_between(hist_dates, res['actual'].values,
                    alpha=0.15, color=col)

    # Forecast connection
    conn_dates = [pd.Timestamp(hist_dates[-1])] + list(future_dates)
    conn_vals  = [res['actual'].values[-1]] + list(res['fc_mean'].values)
    ax.plot(conn_dates, conn_vals,
            color=AMBER, linewidth=2.5, marker='o',
            markersize=5, linestyle='--',
            label='Forecast', zorder=5)

    # Confidence band
    ci_dates = [pd.Timestamp(hist_dates[-1])] + list(future_dates)
    ci_lower = [res['actual'].values[-1]] + list(res['fc_lower'].values)
    ci_upper = [res['actual'].values[-1]] + list(res['fc_upper'].values)
    ax.fill_between(ci_dates, ci_lower, ci_upper,
                    alpha=0.2, color=AMBER,
                    label='80% Confidence')

    # Divider
    ax.axvline(x=pd.Timestamp(hist_dates[-1]),
               color=WHITE, linewidth=1,
               linestyle=':', alpha=0.5)

    # Labels on forecast points
    for d, v in zip(future_dates, res['fc_mean'].values):
        ax.annotate(f'${v/1e6:.2f}M',
                    xy=(d, v), xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', fontsize=7.5,
                    color=AMBER, fontweight='bold')

    leg = ax.legend(facecolor=CARD, fontsize=7, framealpha=0.9)
    for text in leg.get_texts():
        text.set_color(WHITE)
    leg.get_frame().set_edgecolor(GRID)

# 4th panel — comparison bar chart
ax4 = axes[1][1]
ax4.set_facecolor(CARD)
ax4.set_title('3-Month Forecast Comparison by Category',
              color=WHITE, fontsize=12, pad=10, fontweight='bold')

bar_w  = 0.25
x      = np.arange(forecast_steps)
months = [d.strftime('%b %Y') for d in future_dates]

for i, cat in enumerate(categories):
    vals = results[cat]['fc_mean'].values
    bars = ax4.bar(x + i*bar_w, vals,
                   bar_w, label=cat,
                   color=CAT_COLORS[cat], alpha=0.85)
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5000,
                 f'${val/1e6:.2f}M',
                 ha='center', fontsize=7,
                 color=WHITE, fontweight='bold')

ax4.set_xticks(x + bar_w)
ax4.set_xticklabels(months, color=WHITE, fontsize=9)
ax4.tick_params(colors=WHITE)
ax4.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
ax4.set_ylabel('Forecast Revenue', color=WHITE, fontsize=9)
for spine in ax4.spines.values():
    spine.set_color(GRID)
ax4.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.5)
leg4 = ax4.legend(facecolor=CARD, fontsize=8, framealpha=0.9)
for text in leg4.get_texts():
    text.set_color(WHITE)
leg4.get_frame().set_edgecolor(GRID)

plt.tight_layout(pad=3.0)
out = r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\step3_category_forecast.png"
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print(f"\nSaved to: {out}")


# ── 5. EXPORT CATEGORY FORECAST CSV ──────────────────────────
rows = []
for cat in categories:
    res = results[cat]
    for i, date in enumerate(future_dates):
        rows.append({
            'date'         : date,
            'category'     : cat,
            'forecast'     : round(res['fc_mean'].values[i], 2),
            'lower_bound'  : round(res['fc_lower'].values[i], 2),
            'upper_bound'  : round(res['fc_upper'].values[i], 2),
            'model'        : f'ARIMA{res["order"]}',
            'mape'         : round(res['mape'], 2),
            'accuracy'     : round(res['accuracy'], 2)
        })

cat_forecast_df = pd.DataFrame(rows)
cat_out = r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\category_forecast.csv"
cat_forecast_df.to_csv(cat_out, index=False)
print(f"Category forecast exported to: {cat_out}")

# Summary
print("\n" + "=" * 55)
print("CATEGORY FORECAST SUMMARY")
print("=" * 55)
for cat in categories:
    res   = results[cat]
    total = res['fc_mean'].values.sum()
    print(f"{cat:15} | 3M Forecast: ${total:>12,.0f} | "
          f"Accuracy: {res['accuracy']:.1f}%")
