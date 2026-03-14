# Monte Carlo (GBM) for 2023, trained on multiple years
# Plots:
#  - Actual 2023
#  - Daily mean across simulations (avg each day)
#  - Daily median
#  - 5–95% band
# Also saves the plot to a PNG.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= SETTINGS =========
CSV_PATH = "full_history/full_history/AIT.csv"          # change if needed
TARGET_YEAR = 2023
SIMS = 20000
TRADING_DAYS_PER_YEAR = 252
RANDOM_SEED = 0

# Choose ONE training mode:
USE_EXPANDING_WINDOW = False    # True = all years before 2023
# If False, use a fixed window like 2018–2022:
FIXED_TRAIN_YEARS = list(range(2017, 2023))
# ============================

# ---- Load data ----
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower() for c in df.columns]

date_col = "date" if "date" in df.columns else df.columns[0]
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)

price_col = "adj close" if "adj close" in df.columns else "close"
if price_col not in df.columns:
    raise ValueError("CSV must contain 'Adj Close' or 'Close' column.")

df["S"] = pd.to_numeric(df[price_col], errors="coerce")
df = df.dropna(subset=["S"])

df["log_ret"] = np.log(df["S"] / df["S"].shift(1))
df["year"] = df[date_col].dt.year


# TRAIN_YEARS = sorted(df["year"].unique())
# TRAIN_YEARS = [y for y in TRAIN_YEARS if y < TARGET_YEAR]




# ---- Pick training years ----
if USE_EXPANDING_WINDOW:
    train_years = sorted(df["year"].unique())
    train_years = [y for y in train_years if y < TARGET_YEAR]
else:
    train_years = FIXED_TRAIN_YEARS





# ---- Split train/target ----
train = df[df["year"].isin(train_years)].copy()
target = df[df["year"] == TARGET_YEAR].copy()

if train.empty:
    raise ValueError("Training set is empty. Check your train_years.")
if target.empty:
    raise ValueError("Target year not found in CSV.")

rets = train["log_ret"].dropna()
if len(rets) < 30:
    raise ValueError("Not enough training returns to estimate mu/sigma.")

# ---- Estimate mu and sigma (annualized) ----
dt = 1 / TRADING_DAYS_PER_YEAR
m = rets.mean()                # ≈ (mu - 0.5*sigma^2) * dt
v = rets.var(ddof=1)           # ≈ sigma^2 * dt

sigma = np.sqrt(v / dt)        # annualized volatility
mu = (m / dt) + 0.5 * sigma**2 # annualized drift

# ---- Simulation setup ----
last_train_year = max(train_years)
S0 = float(df[df["year"] == last_train_year]["S"].iloc[-1])  # last price in last train year
start_date = df[df["year"] == last_train_year][date_col].iloc[-1]

dates = target[date_col].to_numpy()
actual = target["S"].to_numpy()
n_steps = len(target)

rng = np.random.default_rng(RANDOM_SEED)
Z = rng.standard_normal((n_steps, SIMS))  # Z ~ N(0,1)

drift = (mu - 0.5 * sigma**2) * dt
diff = sigma * np.sqrt(dt)

# ---- Run Monte Carlo ----
log_paths = np.cumsum(drift + diff * Z, axis=0)
paths = S0 * np.exp(log_paths)  # (n_steps, SIMS)

# ---- DAILY stats across simulations (IMPORTANT: axis=1) ----
daily_mean = paths.mean(axis=1)              # avg simulated price each day
daily_median = np.median(paths, axis=1)
p5, p95 = np.percentile(paths, [5, 95], axis=1)

# ---- Sanity checks ----
print("paths.shape:", paths.shape)           # (n_steps, SIMS)
print("daily_mean.shape:", daily_mean.shape) # (n_steps,)
print("Training years:", f"{train_years[0]}–{train_years[-1]}")
print("Start date:", start_date.date(), "S0:", S0)
print("mu (annual):", mu, "sigma (annual):", sigma)

# ---- Plot ----
plt.figure(figsize=(10, 5))
plt.plot(dates, actual, label="Actual 2023")
plt.plot(dates, daily_mean, label="Sim mean (avg each day)")
plt.plot(dates, daily_median, label="Sim median")
plt.fill_between(dates, p5, p95, alpha=0.2, label="Sim 5–95% band")
plt.title(f"Monte Carlo GBM for {TARGET_YEAR} (trained on {train_years[0]}–{train_years[-1]})")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# ---- Save plot ----
out_png = f"aapl_mc_{TARGET_YEAR}_train_{train_years[0]}_{train_years[-1]}_mean.png"
plt.figure(figsize=(10, 5))
plt.plot(dates, actual, label="Actual 2023")
plt.plot(dates, daily_mean, label="Sim mean (avg each day)")
plt.plot(dates, daily_median, label="Sim median")
plt.fill_between(dates, p5, p95, alpha=0.2, label="Sim 5–95% band")
plt.title(f"Monte Carlo GBM for {TARGET_YEAR} (trained on {train_years[0]}–{train_years[-1]})")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig(out_png, dpi=200)
plt.close()
print("Saved plot to:", out_png)
