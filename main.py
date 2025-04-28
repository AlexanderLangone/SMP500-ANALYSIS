import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import sys
sys.dont_write_bytecode = True


from models.cyclical_model import fit_cyclical_model
from models.power_law_model import fit_power_law_model, power_law

from data_downloader import (download_sp500_data, download_gdp_data,
                             download_recession_data)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ── Directories ─────────────────────────────────────────────────────────────
DATA_DIR = "data"
PLOTS_DIR = "plots"
METRICS_DIR = "metrics"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ── Download Data ────────────────────────────────────────────────────────────
download_sp500_data()
download_gdp_data()
download_recession_data()

from analysis.strategy_tester import (calculate_signals, backtest_strategy,
                                      calculate_bollinger_signals,
                                      calculate_momentum_signals,
                                      calculate_triple_ma_signals)

# Load data first
sp_data = pd.read_csv(
    os.path.join(DATA_DIR, "sp500.csv"),
    skiprows=[1, 2],  # Skip the Ticker and empty rows
    index_col=0,
    parse_dates=True)  # Parse index as dates
sp_data = sp_data[['Close']]  # Keep only Close column
sp_data['Close'] = pd.to_numeric(sp_data['Close'], errors='coerce')
sp_data = sp_data.dropna()

# Set default start date based on actual data
_default_start = sp_data.index.min().strftime("%Y-%m-%d")

# Get user input for dates first
print("\n=== Set Plot Data ===")


def get_valid_date(prompt, default_value):
    while True:
        date_input = input(prompt).strip()
        if not date_input:
            return pd.to_datetime(default_value)
        try:
            return pd.to_datetime(date_input, format="%Y-%m-%d")
        except ValueError:
            print(
                "Invalid date format. Please use yyyy-mm-dd format (e.g., 2000-12-31)"
            )


start_date = get_valid_date(
    f"Enter start date (press Enter for {_default_start} or type yyyy-mm-dd): ",
    _default_start)

while True:
    end_date = get_valid_date(
        "Enter end date (press Enter for today or type yyyy-mm-dd): ", "today")
    if end_date > start_date:
        break
    print("End date must be after start date. Please try again.")

# Now filter data based on user's date range
sp_data = sp_data[(sp_data.index >= start_date) & (sp_data.index <= end_date)]

# Test all available strategies
signals_df_ma = calculate_signals(sp_data, rsi_period=None)  # MA only
signals_df_rsi = calculate_signals(sp_data, ma_short=None,
                                   ma_long=None)  # RSI only
signals_df_combined = calculate_signals(sp_data)  # Combined MA + RSI
signals_df_bollinger = calculate_bollinger_signals(sp_data)  # Bollinger Bands
signals_df_momentum = calculate_momentum_signals(
    sp_data)  # Momentum (RSI+MACD)
signals_df_triple_ma = calculate_triple_ma_signals(sp_data)  # Triple MA

results_ma = backtest_strategy(signals_df_ma, "Moving Average Crossover")
results_rsi = backtest_strategy(signals_df_rsi, "RSI")
results_combined = backtest_strategy(signals_df_combined, "Combined MA+RSI")
results_bollinger = backtest_strategy(signals_df_bollinger, "Bollinger Bands")
results_momentum = backtest_strategy(signals_df_momentum,
                                     "Momentum (RSI+MACD)")
results_triple_ma = backtest_strategy(signals_df_triple_ma, "Triple MA")

# All strategies for comparison
strategies = [
    results_ma, results_rsi, results_combined, results_bollinger,
    results_momentum, results_triple_ma
]
best_strategy = max(strategies, key=lambda x: x['Strategy_Return'])
worst_strategy = min(strategies, key=lambda x: x['Strategy_Return'])

while True:
    proj_input = input(
        "Enter projection horizon in years past end date (press Enter for 10 years): "
    ).strip()
    if not proj_input:
        proj_years = 10
        break
    try:
        proj_years = int(proj_input)
        if proj_years <= 0:
            print("Projection years must be greater than 0. Please try again.")
            continue
        break
    except ValueError:
        print("Invalid input. Please enter a valid number.")
proj_end_date = end_date + pd.DateOffset(years=proj_years)

while True:
    investment_input = input(
        "Enter initial investment amount (e.g., $10000 or 10,000 or 10000, press Enter for $100,000): "
    ).strip()
    if not investment_input:
        initial_capital = 100000
        break
    investment_input = investment_input.replace('$', '').replace(',', '')
    try:
        initial_capital = int(investment_input)
        if initial_capital <= 0:
            print(
                "Investment amount must be greater than 0. Please try again.")
            continue
        break
    except ValueError:
        print("Invalid amount format. Please enter a valid number.")

print(f"Using data from {start_date.date()} to {end_date.date()}")
print(f"Forecasting from {end_date.date()} to {proj_end_date.date()}")
print(f"Testing strategies with initial investment of ${initial_capital:,}")

# ── Load & filter S&P 500 ─────────────────────────────────────────────────────
sp = pd.read_csv(os.path.join(DATA_DIR, "sp500.csv"), index_col=0)
sp.index = pd.to_datetime(sp.index, format="%Y-%m-%d", errors="coerce")
sp = sp.sort_index()[["Close"]]
mask = (sp.index >= start_date) & (sp.index <= end_date)
sp = sp.loc[mask].copy()
sp["Close"] = pd.to_numeric(sp["Close"], errors="coerce")
sp.dropna(subset=["Close"], inplace=True)

# compute forward returns and days since start_date
sp["Return1Y"] = sp["Close"].pct_change(252).shift(-252)
sp["Return3Y"] = sp["Close"].pct_change(252 * 3).shift(-252 * 3)
sp["Days"] = (sp.index - start_date).days

# bring Date into column for modeling
sp.index.name = "Date"
sp.reset_index(inplace=True)
sp_model = sp

# ── Fit Models ───────────────────────────────────────────────────────────────
x = sp_model["Days"].values / 3650.0  # decades
y = sp_model["Close"].values

cyc_params = fit_cyclical_model(x, y)
power_params = fit_power_law_model(x, y)

# Simple linear regression using y = mx + b
price_m, price_b = np.polyfit(x, y, 1)  # Store price trend coefficients separately
y_lin_fit = price_m * x + price_b

# in-sample predictions
y_cyc_fit = np.sum([c * x**i for i, c in enumerate(cyc_params)], axis=0)
y_power_fit = power_law(x, *power_params)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


metrics_full = {
    "Cyclical": compute_metrics(y, y_cyc_fit),
    "PowerLaw": compute_metrics(y, y_power_fit),
    "Linear": compute_metrics(y, y_lin_fit),
}

cutoff = sp_model["Date"].max() - pd.DateOffset(years=10)
mask10 = sp_model["Date"] >= cutoff
metrics_10 = {
    name: compute_metrics(y[mask10], preds[mask10])
    for name, preds in
    zip(["Cyclical", "PowerLaw", "Linear"],
        [y_cyc_fit, y_power_fit, y_lin_fit])
}

# Save metrics to CSV
metrics_df_full = pd.DataFrame.from_dict(metrics_full,
                                         orient='index',
                                         columns=['RMSE', 'MAE', 'R2'])
metrics_df_full.to_csv(os.path.join(METRICS_DIR, "metrics_full_history.csv"))

metrics_df_10 = pd.DataFrame.from_dict(metrics_10,
                                       orient='index',
                                       columns=['RMSE', 'MAE', 'R2'])
metrics_df_10.to_csv(os.path.join(METRICS_DIR, "metrics_last_10_years.csv"))

# Print metrics
print("\n" + "=" * 80)
print("MODEL PERFORMANCE METRICS")
print("=" * 80)
print("\n=== Full History Performance ===")
for name, (rmse, mae, r2) in metrics_full.items():
    print(f"{name:12s} RMSE={rmse:,.2f} MAE={mae:,.2f} R²={r2:.4f}")

# print("\n=== Recent Performance (Last 10 Years) ===")
# for name, (rmse, mae, r2) in metrics_10.items():
#     print(f"{name:12s} RMSE={rmse:,.2f} MAE={mae:,.2f} R²={r2:.4f}")

# ── Load & Align GDP & Recession ───────────────────────────────────────────────
gdp = pd.read_csv(os.path.join(DATA_DIR, "gdp.csv"),
                  index_col=0,
                  parse_dates=True)
rec = pd.read_csv(os.path.join(DATA_DIR, "recession.csv"),
                  index_col=0,
                  parse_dates=True).rename(columns={"USREC": "Recession"})

gdp = gdp.reindex(sp_model["Date"]).ffill().bfill()
rec = rec.reindex(sp_model["Date"]).ffill().fillna(0)

df = sp_model.set_index("Date")[["Close", "Return1Y",
                                 "Return3Y"]].join(gdp,
                                                   how="left").join(rec,
                                                                    how="left")

# compute ratio and moving averages
df["Ratio"] = df["Close"] / df["GDP"]
df["StaticMean"] = df["Ratio"].mean()
df["RollingMean5Y"] = df["Ratio"].rolling(window=365 * 5,
                                          min_periods=30).mean()
df["RollingMean1Y"] = df["Ratio"].rolling(window=365, min_periods=30).mean()

# ── BOTH STATIC & EXPANDING REGIMES ────────────────────────────────────────────
df["StaticMu"] = df["Ratio"].mean()
df["StaticSigma"] = df["Ratio"].std()
df["ExpandingMu"] = df["Ratio"].expanding(min_periods=30).mean()
df["ExpandingSigma"] = df["Ratio"].expanding(min_periods=30).std()

df["StaticRegime"] = 0
df["ExpandingRegime"] = 0

df.loc[df["Ratio"] > df["StaticMu"] + df["StaticSigma"], "StaticRegime"] = 1
df.loc[df["Ratio"] < df["StaticMu"] - df["StaticSigma"], "StaticRegime"] = -1
df.loc[df["Ratio"] > df["ExpandingMu"] + df["ExpandingSigma"],
       "ExpandingRegime"] = 1
df.loc[df["Ratio"] < df["ExpandingMu"] - df["ExpandingSigma"],
       "ExpandingRegime"] = -1

# ── Backtest Expanding Regime ─────────────────────────────────────────────────
# Calculate market returns for each regime
bt = df[["Close", "ExpandingRegime"]].copy()
bt = bt.rename(columns={"ExpandingRegime": "Regime"})
bt['Market_Return'] = bt['Close'].pct_change()

summary = bt.groupby("Regime").agg({
    'Market_Return':
    lambda x: ((1 + x).prod() - 1) * 100  # Cumulative return as percentage
}).rename(index={
    -1: "Undervalued",
    0: "Neutral",
    1: "Overvalued"
})

print("\n" + "=" * 80)
print("SMP500/GDP ")
print("Overvalued = Ratio is 1 SD above Expanding Mean ")
print("Undervalued = Ratio is 1 SD below Expanding Mean ")
print("Neutral = Ratio is within 1 SD of Expanding Mean ")
print("=" * 80)

# ── Regression & Recession Models ─────────────────────────────────────────────
# Simple linear regression using y = mx + b
feat = df.dropna(subset=["Return1Y"])
x_vals = feat["Ratio"].values
y_vals = feat["Return1Y"].values

# Calculate slope (m) and intercept (b)
n = len(x_vals)
mean_x = np.mean(x_vals)
mean_y = np.mean(y_vals)
m = np.sum((x_vals - mean_x) * (y_vals - mean_y)) / np.sum((x_vals - mean_x)**2)
b = mean_y - m * mean_x

# Calculate R-squared
y_pred = m * x_vals + b
ss_res = np.sum((y_vals - y_pred) ** 2)
ss_tot = np.sum((y_vals - mean_y) ** 2)
lr_r2 = 1 - (ss_res / ss_tot)

df["Ratio_Lag1Y"] = df["Ratio"].shift(252)
rc = df.dropna(subset=["Ratio_Lag1Y"])
# Simple threshold-based classification
ratio_threshold = rc["Ratio_Lag1Y"].mean() + rc["Ratio_Lag1Y"].std()
predictions = (rc["Ratio_Lag1Y"] > ratio_threshold).astype(int)
log_acc = np.mean(predictions == rc["Recession"])

# Save regression performance
with open(os.path.join(METRICS_DIR, "model_performance.txt"), "w") as f:
    f.write(f"LR R^2 on 1Y return: {lr_r2:.3f}\n")
    f.write(f"Logistic accuracy: {log_acc:.3f}\n")

# ── Drawdown Analysis ─────────────────────────────────────────────────────────
df["MaxClose"] = df["Close"].cummax()
df["Drawdown"] = df["Close"] / df["MaxClose"] - 1
dd_exp = df.groupby("ExpandingRegime")["Drawdown"].min() \
           .rename(index={-1:"Undervalued",0:"Neutral",1:"Overvalued"})

# Save drawdown summary
(dd_exp * 100).round(2).to_csv(
    os.path.join(METRICS_DIR, "max_drawdown_by_regime.csv"))

print("\n=== Maximum Drawdowns ===")
dd_frame = (dd_exp * 100).round(2)
print("Undervalued: {:.2f}%, Neutral: {:.2f}%, Overvalued: {:.2f}%".format(
    dd_frame['Undervalued'], dd_frame['Neutral'], dd_frame['Overvalued']))

# ── Forecast & Model Comparison Plot ──────────────────────────────────────────
future = pd.date_range(start=end_date, end=proj_end_date, freq="D")
fdays_array = np.array((future - start_date).days / 3650.0)

y_c_f = np.sum([c * fdays_array**i for i, c in enumerate(cyc_params)], axis=0)
y_p_f = power_law(fdays_array, *power_params)

# Calculate linear forecast using price trend coefficients
y_l_f = price_m * fdays_array + price_b

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df["Close"], label="Actual", alpha=0.5)
ax.plot(sp_model["Date"], y_cyc_fit, label="Cyc Fit")
ax.plot(sp_model["Date"], y_power_fit, label="Power Fit")
ax.plot(sp_model["Date"], y_lin_fit, label="Linear Fit")

# forecasts for each fit
ax.plot(future, y_c_f, "--", label="Cyc Forecast")
ax.plot(future, y_p_f, "--", label="Power Forecast")
ax.plot(future, y_l_f, "--", label="Linear Forecast")

ymin, ymax = ax.get_ylim()
ax.fill_between(df.index,
                ymin,
                ymax,
                where=df["Recession"] == 1,
                facecolor="gray",
                alpha=0.3)
ax.set_title("S&P 500 Models & Forecast with Recessions")
ax.set_xlabel("Date")
ax.set_ylabel("Index Level")
ymax = max(max(y_c_f), max(y_p_f), max(y_l_f)) * 1.1  # Add 10% padding
ax.set_ylim(0, ymax)
ax.legend(loc="upper left")
fig.autofmt_xdate()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_with_forecast.png"),
            dpi=300)
plt.close()

# normalized x for prediction
pred_days = (end_date - start_date).days
pred_x = pred_days / 3650.0

# model predictions
cyc_pred = sum(c * pred_x**i for i, c in enumerate(cyc_params))
power_pred = power_law(pred_x, *power_params)
lin_pred = m * pred_x + b

# last observed actual
last_price = sp_model["Close"].iloc[-1]
delta_days = (pd.Timestamp(pred_days) - end_date).days
years_ahead = delta_days / 365.0 if delta_days > 0 else 0.0

# ROI calculations
# Portfolio Growth Analysis
from analysis.portfolio_growth import analyze_portfolio_growth

portfolio_results = analyze_portfolio_growth(df,
                                             initial_capital=initial_capital)