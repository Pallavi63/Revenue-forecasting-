# ============================================================
# Revenue Forecasting — Step 1 Charts (JUPYTER FIXED)
# ============================================================

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ── LOAD DATA ────────────────────────────────────────────────
monthly = pd.read_csv(
    r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\monthly_revenue.csv",
    parse_dates=['date'])
monthly = monthly.sort_values('date').reset_index(drop=True)
monthly['mom_growth'] = monthly['total_revenue'].pct_change() * 100

category_monthly = pd.read_csv(
    r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\monthly_revenue_by_category.csv",
    parse_dates=['date'])
category_monthly = category_monthly.sort_values(['product_category', 'date'])

# ── COLORS ───────────────────────────────────────────────────
CYAN  = '#00B4D8'
AMBER = '#F4A825'
GREEN = '#2ECC71'
RED   = '#E74C3C'
WHITE = '#FFFFFF'
BG    = '#1B2631'
CARD  = '#212F3D'
GRID  = '#3D5166'

# ── HELPER ───────────────────────────────────────────────────
def style_axis(ax, title, ylabel):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=WHITE, fontsize=12,
                 pad=10, fontweight='bold')
    ax.set_ylabel(ylabel, color=WHITE, fontsize=9)
    ax.set_xlabel('Month', color=WHITE, fontsize=9)
    ax.tick_params(colors=WHITE, labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(),
             rotation=45, ha='right', color=WHITE)
    ax.yaxis.label.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis='y', color=GRID, linewidth=0.5, alpha=0.6)

# ── FIGURE ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.patch.set_facecolor(BG)
fig.suptitle('Revenue Time Series — Exploratory Analysis',
             fontsize=16, color=WHITE, fontweight='bold')

# ── CHART 1 — Revenue Trend ──────────────────────────────────
ax1 = axes[0, 0]
style_axis(ax1, 'Monthly Revenue Trend', 'Revenue')
ax1.plot(monthly['date'], monthly['total_revenue'],
         color=CYAN, linewidth=2.5, marker='o',
         markersize=6, zorder=5, label='Revenue')
ax1.fill_between(monthly['date'], monthly['total_revenue'],
                 alpha=0.2, color=CYAN)
for i, row in monthly.iterrows():
    ax1.annotate(
        f"${row['total_revenue']/1e6:.1f}M",
        xy=(row['date'], row['total_revenue']),
        xytext=(0, 8), textcoords='offset points',
        ha='center', fontsize=6.5, color=WHITE)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
ax1.set_ylim(0, monthly['total_revenue'].max() * 1.3)
ax1.tick_params(axis='y', colors=WHITE)

# ── CHART 2 — Order Volume ───────────────────────────────────
ax2 = axes[0, 1]
style_axis(ax2, 'Monthly Order Volume', 'Orders')
bars = ax2.bar(monthly['date'], monthly['total_orders'],
               color=AMBER, alpha=0.85, width=22, zorder=5)
for bar, val in zip(bars, monthly['total_orders']):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f'{int(val):,}',
        ha='center', fontsize=6.5, color=WHITE)
ax2.set_ylim(0, monthly['total_orders'].max() * 1.2)
ax2.tick_params(axis='y', colors=WHITE)

# ── CHART 3 — Category Revenue ───────────────────────────────
ax3 = axes[1, 0]
style_axis(ax3, 'Monthly Revenue by Category', 'Revenue')
cat_colors = {'Bikes': CYAN, 'Accessories': AMBER, 'Clothing': GREEN}
for cat, color in cat_colors.items():
    sub = category_monthly[category_monthly['product_category'] == cat]
    ax3.plot(sub['date'], sub['total_revenue'],
             color=color, linewidth=2.5, marker='o',
             markersize=5, label=cat, zorder=5)
    ax3.fill_between(sub['date'], sub['total_revenue'],
                     alpha=0.08, color=color)
ax3.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
ax3.tick_params(axis='y', colors=WHITE)
leg = ax3.legend(facecolor=CARD, fontsize=8, framealpha=0.9)
for text in leg.get_texts():
    text.set_color(WHITE)
leg.get_frame().set_edgecolor(GRID)

# ── CHART 4 — MoM Growth ─────────────────────────────────────
ax4 = axes[1, 1]
style_axis(ax4, 'Month-over-Month Growth %', 'Growth %')
bar_colors = [GREEN if x >= 0 else RED
              for x in monthly['mom_growth'].fillna(0)]
bars2 = ax4.bar(monthly['date'], monthly['mom_growth'].fillna(0),
                color=bar_colors, alpha=0.85, width=22, zorder=5)
ax4.axhline(y=0, color=WHITE, linewidth=1,
            linestyle='--', alpha=0.5)
for bar, val in zip(bars2, monthly['mom_growth'].fillna(0)):
    offset = 1.5 if val >= 0 else -4
    ax4.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + offset,
        f'{val:.1f}%',
        ha='center', fontsize=6.5, color=WHITE)
ax4.tick_params(axis='y', colors=WHITE)

# ── SAVE & SHOW ──────────────────────────────────────────────
plt.tight_layout(pad=2.5)

output_path = r"C:\Users\burug\OneDrive\Desktop\projects\revenue forecasting\step1_eda_charts.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')

plt.show()
print(f"\nSaved to: {output_path}")
