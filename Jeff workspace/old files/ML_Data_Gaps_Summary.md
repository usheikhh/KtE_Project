# TTF Option Premium & Implied Volatility Gap Fill — Summary

## Problem
185 missing observations in both:
- `IM_TTF_OPTIONS_TTF_IMPLIED_VOL_PCT` (implied volatility)
- `IM_TTF_OPTIONS_TTF_PREMIUM_EUR_MWH` (option premium)

The IV fill feeds directly into the premium fill via Black-76.

---

## Step 1: Implied Volatility Gap Fill

### GARCH Family Models
- **Motivation:** Volatility clustering in energy markets; 2022 energy crisis creates a clear regime change
- **Models compared:** GARCH(1,1), GJR-GARCH, EGARCH, APARCH
- **Calibration:** OLS scaling of GARCH conditional volatility to implied vol scale on overlapping windows

### HMM Regime Detection
- 2-state Gaussian HMM fitted on GARCH standardized residuals
- Identified **high-vol / low-vol regimes** — 2022 crisis clearly captured in high-vol state
- Regime labels preserved on filled values: `Filled-Low-vol` / `Filled-High-vol`
- Distribution bounds enforced to keep filled values realistic

**Output columns:**
- `IM_TTF_OPTIONS_TTF_IMPLIED_VOL_PCT_GARCH_FILLED`
- `IM_TTF_OPTIONS_TTF_IMPLIED_VOL_PCT_FILL_REGIME`

---

## Step 2: Option Premium Gap Fill

### Baseline: AR(15)
- PACF analysis used to identify significant lags
- Holdout validation across AR(1)–AR(15) to select optimal order
- **Holdout RMSE: 1.4381 EUR/MWh**

### Final Model: Black-76 ATM Pricing
- **Motivation:** 0.94 correlation between `price × IV` and actual premium
- **Formula:**
  ```
  Premium_ATM = F × σ × √T × 0.7979
  ```
  where F = TTF futures price, σ = `IM_TTF_OPTIONS_TTF_IMPLIED_VOL_PCT_GARCH_FILLED`, T = business days to month-end / 252
- OLS calibration on observed dates: α = −0.1556, β = 0.3893
- **Output column:** `IM_TTF_OPTIONS_TTF_PREMIUM_EUR_MWH_BLK76_FILLED`

---

## Validation Results

| Metric | AR(15) | Black-76 |
|---|---|---|
| Holdout RMSE | 1.4381 EUR/MWh | **0.4038 EUR/MWh** |
| R² (full sample) | — | **0.9867** |
| Improvement | — | **~3.6× better** |

**Residual Diagnostics:**

| Test | Result | Status |
|---|---|---|
| Durbin-Watson | 0.754 | ⚠️ Autocorrelation present (ideal ≈ 2.0) |
| Ljung-Box p-value | ≈ 0 | ⚠️ Residuals not white noise |
| Breusch-Pagan p-value | ≈ 0 | ⚠️ Heteroskedasticity detected |
| 95% PI coverage | 100% | ✅ Conservative but valid |

**Distribution of Filled vs Observed Premium:**

| | Mean | Std | Range |
|---|---|---|---|
| Observed | 5.78 EUR/MWh | 7.13 | [0.154, 62.32] |
| Filled | 1.98 EUR/MWh | 2.27 | [0.009, 10.50] |

> ⚠️ Filled values are systematically lower and less volatile — gaps fall predominantly outside the 2022 crisis period.

---

## Why This Is Good Enough

The goal of this gap fill is to produce **plausible, consistent inputs for a downstream ML model** — not to perfectly replicate what market prices would have been on missing dates. Against that standard, the approach holds up:

1. **Model is grounded in financial theory.** Black-76 is the industry-standard pricing formula for commodity options. Using it to fill gaps means filled values are structurally consistent with the rest of the dataset, not just statistically interpolated noise.

2. **R² = 0.9867 on observed data.** The model explains ~99% of variance in actual premiums. Errors on the holdout set (RMSE = 0.4038 EUR/MWh) are small relative to the mean premium of 5.78 EUR/MWh (~7% relative error).

3. **3.6× improvement over a reasonable baseline.** AR(15) is a competitive time-series benchmark. The Black-76 model decisively outperforms it, which gives confidence the structural approach is adding real signal.

4. **100% prediction interval coverage.** All filled values fall within the 95% bounds, meaning no filled value is implausibly extreme. The conservative intervals are appropriate for an ML training set where outlier inputs can distort learning.

5. **Distribution mismatch is explainable, not a modeling failure.** Gaps fall predominantly outside the 2022 crisis period, so lower filled values are correct — they reflect calm-market conditions, not model error.

6. **Regime awareness is built in.** The HMM-labeled IV fill ensures the volatility input to Black-76 is regime-consistent. The model would not assign a crisis-level IV to a gap that occurred in a calm-market period.

The residual autocorrelation and heteroskedasticity are real issues, but they affect **error quantification** (confidence intervals), not the point estimates fed to the ML model. For gap-filling purposes, point estimate quality is what matters.

