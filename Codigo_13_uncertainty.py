# Codigo 13 - ok =====================================================
# === SST July — Uncertainty quantification for the 2025 projection | MO/L3m ===
# This cell is the response to the reviewers. It reproduces, from the same
# localized MODIS L3m July time series used in Codigo 12, the following:
#   (1) OLS slope significance via t-test (with 95% CI for the slope)
#   (2) 95% confidence interval for the mean ŷ(2025)
#   (3) 95% prediction interval for a single observed value in July 2025
#   (4) Probability of exceeding decision thresholds in July 2025
#       (P(SST>29.5°C), P(SST>30°C), P(SST>30.5°C), P(SST>31°C))
#   (5) Mann–Kendall non-parametric trend test + Theil–Sen slope estimator
#
# The numbers produced here are the ones reported in Annex I, Sections
# 1.4.1, 1.4.2, and 1.4.3 of the revised manuscript.

MONTH    = 7      # July
TARGET_Y = 2025   # year to project to
CSV_DIR  = "/content/drive/MyDrive/Colab Notebooks/SST_Singapore_Strait/Dados/MO/L3m/csv_export"

# --- Mount Google Drive (Colab) ---
try:
    from google.colab import drive
    print("↪ Mounting Google Drive...")
    drive.mount('/content/drive')
except Exception:
    print("↪ Non-Colab environment (skipping mount).")

import os, glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
from datetime import datetime

# -------------------------------------------------------------------------
# 1) Load monthly CSVs (same loader logic as Codigo 06/07/08/09/12)
# -------------------------------------------------------------------------
assert os.path.isdir(CSV_DIR), f"Directory not found: {CSV_DIR}"
files = sorted(glob.glob(os.path.join(CSV_DIR, "*.csv.gz"))) + \
        sorted(glob.glob(os.path.join(CSV_DIR, "*.csv")))
assert files, "No CSV files found."

def load_one_csv_monthly(path):
    head = pd.read_csv(path, nrows=0)
    cols = list(head.columns)
    sst_col = next((c for c in ["sst","sst4","sea_surface_temperature"] if c in cols), None)
    if sst_col is None:
        sst_like = [c for c in cols if "sst" in c.lower()]
        sst_col = sst_like[0] if sst_like else None
    if sst_col is None:
        return None
    if "date" in cols:
        df = pd.read_csv(path, usecols=["date", sst_col])
        dt = pd.to_datetime(df["date"], errors="coerce")
        month_date = pd.to_datetime(dt.dt.to_period("M").dt.to_timestamp())
        out = pd.DataFrame({
            "year": month_date.dt.year,
            "month": month_date.dt.month,
            "sst": pd.to_numeric(df[sst_col], errors="coerce")
        })
    else:
        needed = [c for c in ["year","month","datetime_iso"] if c in cols]
        if not needed: return None
        if {"year","month"}.issubset(needed):
            dft = pd.read_csv(path, usecols=["year","month"])
            y = pd.to_numeric(dft["year"], errors="coerce")
            m = pd.to_numeric(dft["month"], errors="coerce")
            out = pd.DataFrame({"year": y, "month": m})
        else:
            dft = pd.read_csv(path, usecols=["datetime_iso"])
            dt = pd.to_datetime(dft["datetime_iso"], errors="coerce")
            out = pd.DataFrame({"year": dt.dt.year, "month": dt.dt.month})
        sst_series = pd.read_csv(path, usecols=[sst_col])[sst_col]
        out["sst"] = pd.to_numeric(sst_series, errors="coerce")
    out = out.dropna(subset=["year","month","sst"]).astype({"year": int, "month": int})
    return out[["year","month","sst"]]

dfs = []
for f in files:
    try:
        dfx = load_one_csv_monthly(f)
        if dfx is not None and not dfx.empty:
            dfs.append(dfx)
    except Exception:
        pass
assert dfs, "Failed to load monthly CSVs."
data = pd.concat(dfs, ignore_index=True)

# -------------------------------------------------------------------------
# 2) Build the July one-point-per-year series
# -------------------------------------------------------------------------
month_data    = data[data["month"] == MONTH].copy()
per_month     = month_data.groupby(["year","month"], as_index=False)["sst"].mean()
annual_series = per_month.groupby("year", as_index=False)["sst"].mean().sort_values("year")
annual_series = annual_series.reset_index(drop=True)

x = annual_series["year"].astype(float).values
y = annual_series["sst"].astype(float).values
n = len(x)
assert n >= 5, "Not enough July years to fit a trend."

print(f"\n▶ July series: n = {n}  ({int(x.min())}–{int(x.max())})")

# -------------------------------------------------------------------------
# 3) OLS regression with full uncertainty machinery
# -------------------------------------------------------------------------
# slope, intercept, R² from polyfit (matches Codigo 12 exactly)
a, b = np.polyfit(x, y, 1)
y_hat = a * x + b
ss_res = np.sum((y - y_hat) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1.0 - ss_res / ss_tot

# residual standard error (RMSE with n-2 degrees of freedom)
sigma_res = np.sqrt(ss_res / (n - 2))

# Sxx and standard errors of a, b
Sxx = np.sum((x - x.mean()) ** 2)
SE_a = sigma_res / np.sqrt(Sxx)
SE_b = sigma_res * np.sqrt(1.0 / n + x.mean() ** 2 / Sxx)

# t-test on the slope
t_a   = a / SE_a
p_a   = 2 * (1 - stats.t.cdf(abs(t_a), df=n - 2))
tcrit = stats.t.ppf(0.975, df=n - 2)
CI_a  = (a - tcrit * SE_a, a + tcrit * SE_a)

print("\n=== OLS regression for July ===")
print(f"  slope  a  : {a:.6f} °C/year")
print(f"  intercept b: {b:.4f} °C  (algebraic artefact at x=0)")
print(f"  R²        : {r2:.4f}")
print(f"  σ_res     : {sigma_res:.4f} °C")
print(f"  SE(a)     : {SE_a:.6f}    t={t_a:.3f}    p={p_a:.4f}    df={n-2}")
print(f"  95% CI(a) : [{CI_a[0]:.5f}, {CI_a[1]:.5f}] °C/year")

# -------------------------------------------------------------------------
# 4) Projection for TARGET_Y with confidence and prediction intervals
# -------------------------------------------------------------------------
x_new = float(TARGET_Y)
y_new = a * x_new + b

# SE of the predicted mean (confidence interval band)
SE_yhat = sigma_res * np.sqrt(1.0 / n + (x_new - x.mean()) ** 2 / Sxx)
CI_yhat = (y_new - tcrit * SE_yhat, y_new + tcrit * SE_yhat)

# SE of a single new observation (prediction interval)
SE_pred = sigma_res * np.sqrt(1.0 + 1.0 / n + (x_new - x.mean()) ** 2 / Sxx)
PI_obs  = (y_new - tcrit * SE_pred, y_new + tcrit * SE_pred)

print(f"\n=== Projection for July {TARGET_Y} ===")
print(f"  point estimate ŷ        : {y_new:.3f} °C")
print(f"  95% CI for the mean     : [{CI_yhat[0]:.3f}, {CI_yhat[1]:.3f}] °C")
print(f"  95% prediction interval : [{PI_obs[0]:.3f}, {PI_obs[1]:.3f}] °C  (USE THIS for event-risk talk)")

# -------------------------------------------------------------------------
# 5) Threshold exceedance probabilities under the model (Gaussian residual
#    assumption around the fitted line, with prediction-interval std error)
# -------------------------------------------------------------------------
def p_exceed(threshold, mu, sigma):
    return 1.0 - stats.norm.cdf(threshold, loc=mu, scale=sigma)

thresholds = [29.0, 29.5, 30.0, 30.5, 31.0]
print(f"\n=== Threshold exceedance probabilities for July {TARGET_Y} ===")
for th in thresholds:
    p = p_exceed(th, y_new, SE_pred)
    marker = ""
    if th == 30.0: marker = "  (event temperature recorded in 2025)"
    if th == 31.0: marker = "  (World Aquatics operational threshold)"
    print(f"  P(SST > {th:.1f} °C) = {p:.3f}  ({p*100:.1f}%){marker}")

# -------------------------------------------------------------------------
# 6) Mann–Kendall non-parametric trend test + Theil–Sen slope
# -------------------------------------------------------------------------
S = 0
for i in range(n - 1):
    S += np.sum(np.sign(y[i + 1:] - y[i]))
var_S = n * (n - 1) * (2 * n + 5) / 18.0
if S > 0:   Z = (S - 1) / np.sqrt(var_S)
elif S < 0: Z = (S + 1) / np.sqrt(var_S)
else:       Z = 0.0
p_MK = 2 * (1 - stats.norm.cdf(abs(Z)))

# Theil–Sen slope
slopes_all = []
for i in range(n - 1):
    slopes_all.extend((y[i + 1:] - y[i]) / (x[i + 1:] - x[i]))
sen_slope = np.median(slopes_all)

print(f"\n=== Non-parametric robustness check ===")
print(f"  Mann–Kendall S       : {int(S)}")
print(f"  Mann–Kendall Z       : {Z:.3f}")
print(f"  Mann–Kendall p-value : {p_MK:.4f}  (two-sided)")
print(f"  Theil–Sen slope      : {sen_slope:.6f} °C/year")

# -------------------------------------------------------------------------
# 7) Plot: data + fit + 95% CI band + 95% PI band + projection point
# -------------------------------------------------------------------------
x_grid = np.linspace(x.min(), TARGET_Y, 200)
y_grid = a * x_grid + b
se_y_grid = sigma_res * np.sqrt(1.0 / n + (x_grid - x.mean()) ** 2 / Sxx)
se_p_grid = sigma_res * np.sqrt(1.0 + 1.0 / n + (x_grid - x.mean()) ** 2 / Sxx)

x_grid_date = pd.to_datetime(pd.Series(x_grid).round().astype(int).astype(str) + "-01-01")
x_data_date = pd.to_datetime(annual_series["year"].astype(str) + "-01-01")
x_pred_date = pd.to_datetime(f"{TARGET_Y}-01-01")

plt.figure(figsize=(10, 5.5))
plt.fill_between(x_grid_date, y_grid - tcrit * se_p_grid, y_grid + tcrit * se_p_grid,
                 alpha=0.15, label="95% prediction interval")
plt.fill_between(x_grid_date, y_grid - tcrit * se_y_grid, y_grid + tcrit * se_y_grid,
                 alpha=0.30, label="95% CI of the mean")
plt.plot(x_grid_date, y_grid, linewidth=1.5, label=f"OLS fit (a = {a:.4f} °C/yr)")
plt.plot(x_data_date, y, "o", label="July annual means (observed)")
plt.plot(x_pred_date, y_new, "*", markersize=14,
         label=f"July {TARGET_Y}: {y_new:.2f} °C  (PI 95%: {PI_obs[0]:.2f}–{PI_obs[1]:.2f} °C)")

# Reference lines
plt.axhline(31.0, linestyle="--", linewidth=1, alpha=0.6,
            label="World Aquatics threshold (31 °C)")
plt.axhline(30.0, linestyle=":",  linewidth=1, alpha=0.6,
            label="Event temperature recorded (30 °C)")

plt.title(f"July SST in the Singapore Strait — OLS fit, 95% CI and 95% PI, projection to {TARGET_Y}")
plt.xlabel("Year")
plt.ylabel("July SST (°C)")
plt.grid(True, alpha=0.4)
ax = plt.gca()
ax.xaxis.set_major_locator(YearLocator(base=2))
ax.xaxis.set_major_formatter(DateFormatter("%Y"))
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()

fig_dir = os.path.join(CSV_DIR, "fig")
os.makedirs(fig_dir, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
base = f"sst_july_uncertainty_projection_{TARGET_Y}_{ts}"
png = os.path.join(fig_dir, base + ".png")
svg = os.path.join(fig_dir, base + ".svg")
plt.savefig(png, dpi=150)
plt.savefig(svg)
plt.show()

# -------------------------------------------------------------------------
# 8) Export a tidy CSV with everything the revised Annex I cites
# -------------------------------------------------------------------------
summary_rows = [
    ("n_years",                              n),
    ("slope_a_deg_per_year",                 a),
    ("intercept_b_deg",                      b),
    ("R_squared",                            r2),
    ("sigma_residual_deg",                   sigma_res),
    ("SE_slope",                             SE_a),
    ("t_slope",                              t_a),
    ("p_value_slope_two_sided",              p_a),
    ("CI95_slope_low",                       CI_a[0]),
    ("CI95_slope_high",                      CI_a[1]),
    ("projection_year",                      TARGET_Y),
    ("projection_point_estimate_deg",        y_new),
    ("CI95_mean_low_deg",                    CI_yhat[0]),
    ("CI95_mean_high_deg",                   CI_yhat[1]),
    ("PI95_obs_low_deg",                     PI_obs[0]),
    ("PI95_obs_high_deg",                    PI_obs[1]),
    ("P_exceed_29_5_deg",                    p_exceed(29.5, y_new, SE_pred)),
    ("P_exceed_30_0_deg",                    p_exceed(30.0, y_new, SE_pred)),
    ("P_exceed_30_5_deg",                    p_exceed(30.5, y_new, SE_pred)),
    ("P_exceed_31_0_deg",                    p_exceed(31.0, y_new, SE_pred)),
    ("MannKendall_S",                        int(S)),
    ("MannKendall_Z",                        Z),
    ("MannKendall_p_value_two_sided",        p_MK),
    ("TheilSen_slope_deg_per_year",          sen_slope),
]
summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
csv_out = os.path.join(fig_dir, f"sst_july_uncertainty_summary_{TARGET_Y}_{ts}.csv")
summary_df.to_csv(csv_out, index=False)

# Also export the raw July annual series so reviewers can replicate exactly
series_out = os.path.join(fig_dir, f"sst_july_annual_means_for_regression_{ts}.csv")
annual_series.to_csv(series_out, index=False)

print(f"\n✓ Figure saved: {png}")
print(f"✓ Summary CSV : {csv_out}")
print(f"✓ Raw series  : {series_out}")
print("\n✅ Done — these are the numbers reported in Annex I §§ 1.4.1–1.4.3.")
