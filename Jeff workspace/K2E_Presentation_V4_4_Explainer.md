# K2 Energy V4.4 / Track 4 — Slide-by-Slide Explainer (Plain English)

> A companion to `K2E_Presentation_V4_4.html`. Each section corresponds to one slide and is written for someone who has not seen the V4.3 deck and is not a quant. Technical terms are introduced once and then reused. If you only read one section, read **Slide 5 (the bootstrap idea)** and **Slide 17 (the bottom line)**.

---

## Background you need before slide 1

The K2 Energy / Fordham project tries to forecast price moves in three European energy markets:

- **TTF** — Dutch natural-gas futures (the European gas benchmark)
- **POWER** — German baseload electricity futures
- **EUA** — EU carbon-emission allowances

For each market we forecast at three horizons: **1, 3, and 5 trading days** ahead. That gives us 3 markets × 3 horizons = **9 cells** of forecasts.

Two earlier versions of the model already exist:

- **V4.2** — built two predictors per cell: an **L1** (linear regression with shrinkage, called "Ridge") and an **L3** (a regime-switching model that assumes the market alternates between calm and volatile states).
- **V4.3** — added two pieces of discipline on top of V4.2:
    1. A **policy gate** that, per cell, picks the best of {do-nothing baseline ("naive"), L1, L3} — but rejects any model that's actually worse than naive.
    2. A second deliverable: a **sign classifier** (logistic regression) that predicts only direction (up/down), not magnitude.

V4.3 ended with two RMSE wins (TTF h=1 L3, POWER h=1 L3) and two directional wins (EUA h=3 sign, EUA h=5 sign).

**This deck (V4.4) does NOT add a new model.** It stress-tests V4.3's claims using three diagnostic procedures, collectively called "Track 4." That's the entire purpose.

---

## Slide 1 — Title

We're presenting **Track 4: robustness and honest-disclosure diagnostics**. The deliverable is not "another model" but rather the answer to: *"For each thing V4.3 claimed to have found, how confident should we be?"*

Three diagnostic tools are introduced (bootstrap CIs, per-regime slice, rolling-window CV). All run through a wrapper file `K2E_Model_V4_4.py` that re-uses V4.3 unchanged.

---

## Slide 2 — What V4.4 actually is

V4.4 was originally going to be V4.3 + new Bloomberg data fields (positioning data from ESMA, options-volatility "surface"). Both attempts **regressed** vs V4.3:

- The new positioning fields gave information that already lived in features V4.3 was using — so they added noise, not signal.
- The vol-surface attempt hit a 2-year hole in Bloomberg's data — a data-pull problem, not a model problem.

So V4.4 was repurposed as a diagnostics layer. Four substeps:

1. **Safe-fit wrapper** — defensive plumbing so a single failed model fit doesn't crash the whole pipeline.
2. **Bootstrap confidence intervals** — quantify how lucky vs skilled each "win" is.
3. **Per-regime slice** — for the regime-switching model, ask which regime (calm or volatile) is actually carrying each win.
4. **Rolling-window CV** — does the win still appear if the model is only allowed to learn from the most recent 2 years?

> **The point**: V4.3 reported things like "this cell beats baseline by 9%, p=0.028." Track 4 asks: *would we get the same conclusion under different sensible setups?* That's how a serious quant team validates a model before letting traders use it.

---

## Slide 3 — Track 4 architecture overview

Just shows the four substeps as four boxes flowing left to right. The key implementation detail is that all four substeps **re-use V4.3's chosen settings**. We do not re-tune anything. We change **only the evaluation protocol**, which means anything that breaks here was always broken — V4.3's single walk-forward pass just couldn't see it.

The driver function `train_all_v44()` returns a 10-tuple of objects: V4.3's seven outputs plus the three new diagnostic dictionaries (`boot`, `regime`, `rolling`). One Python command reproduces the entire disclosure pack.

---

## Slide 4 — Step #1: Safe-fit wrapper

**The problem** in plain English: V4.3's regime-switching model has two parts: (a) the cross-validation step that scores it, and (b) a final "fit on all the data" step we use to extract regime probabilities. In rare cases the final fit can hit a numerical edge case and throw an error. Before this fix, that error would crash the whole 9-cell pipeline run.

**The fix** is a 10-line wrapper that catches the exception, logs it, returns `None`, and lets the pipeline continue. The cross-validation metrics are already separately computed, so we don't lose anything important per cell.

**Why it matters today**: during the rolling-CV run for Track 4, the **EUA h=1** cell hit exactly this kind of failure on the full-history fit. With the wrapper in place, the pipeline:

- Logged the error (so we know it happened)
- Kept EUA h=1's CV metrics
- Continued through the remaining 8 cells

Without the wrapper, the entire 30-minute Track 4 run would have died on cell 1. So this is the unsexy plumbing that lets Tracks #2/#3/#4 actually finish.

---

## Slide 5 — Step #2: Bootstrap idea (methodology)

This is the most important conceptual slide in the deck.

**The problem** V4.3 reports single-number metrics like "RMSE 0.04172, DM stat −2.20, p=0.028." But each of those numbers comes from one specific way of cutting up the historical data into training and test windows. If we re-cut the data slightly differently, would we get the same numbers? Probably not exactly. The question is: how *similar* would the numbers be?

**Bootstrap** is a statistical trick that answers this. We take the test predictions we already have, randomly resample them many times (1000 reps here), recompute the metric on each resample, and look at the spread. The resulting **90% confidence interval** says: "if we re-sampled the test windows over and over, the metric would land in this range 90% of the time."

**Why "block" bootstrap**: when we forecast "5 days ahead," yesterday's prediction and today's prediction overlap — they're not independent. Plain row-wise bootstrap underestimates noise because it ignores this overlap. Block bootstrap resamples short *blocks* of consecutive observations (here, length `h+1`), which preserves the overlap structure.

**How to read the CIs**:

- If the CI for "model RMSE − naive RMSE" is entirely below zero → the model genuinely beats naive at the 90% level.
- If the CI crosses zero → maybe the model is better, maybe not. Don't bet money on it yet.

**What bootstrap does NOT prove**: it does not prove the edge will hold on next year's data. It proves that the win we observed this year wasn't a single noisy draw. The "out-of-sample on new data" question is what rolling-window CV (Step #4) addresses.

---

## Slide 6 — Bootstrap RMSE results

A big table with one row per (cell, layer). The columns we care most about are the **CI columns** for "vs naive %" and "DM stat."

**How to scan the table**:

- Green rows = the entire CI is on the favorable side (model wins).
- Red rows = the entire CI is on the unfavorable side (model is worse than naive — and we're confident about that).
- Amber/uncolored = CI crosses zero (inconclusive).

**The headline**: only **TTF h=1 L3** has both a vs-naive % CI strictly below zero (`[−1.8%, −0.1%]`) and a DM CI strictly negative (`[−3.63, −0.21]`). That's the only RMSE cell that survives the bootstrap. **POWER h=1 L3**, which V4.3 advertised as RMSE −12% vs naive, has a CI that just-barely crosses zero (`[−3.8%, +0.5%]`) — promising but unproven.

Several cells V4.3 already vetoed (POWER h=5 L3, EUA h=3 L3, EUA h=5 L3, TTF h=3 L3, TTF h=5 L3) show their CI strictly *worse* than naive. The bootstrap confirms V4.3's veto wasn't being conservative — those cells really were broken.

---

## Slide 7 — Bootstrap sign-classifier results

Same table format but for the sign classifier (the up/down predictor). Columns: directional accuracy (DA), Pesaran-Timmermann statistic (PT — a directional-skill test), expected value per trade (EV), and a crude annualised Sharpe ratio.

**The headline**: only **EUA h=3** has all four CIs (DA, PT, EV, Sharpe) strictly on the favorable side. That's the only sign cell that's bootstrap-clean.

**EUA h=5** (one of V4.3's claimed directional wins) has a PT CI that just-crosses zero (`[−0.52, +4.27]`). Promising but on the edge.

**POWER h=1** has the highest Sharpe point estimate (+1.09) but every CI crosses zero — a few large-magnitude trades may be doing all the work. Worth investigating, not deploying.

---

## Slide 8 — Bootstrap headline

Three columns sorting all 9 cells × 2 layers into:

- **Bootstrap-robust at 90%** (2 cells): TTF h=1 L3 RMSE, EUA h=3 sign.
- **Promising but CI crosses zero** (~4 cells): POWER h=1 RMSE, EUA h=5 sign, TTF h=3/h=5 sign.
- **Bootstrap confirms broken** (~5 cells): all the cells V4.3 already routed to naive.

**The honest revision** of V4.3's headline: V4.3 said "two RMSE wins, two directional wins." The strict bootstrap version is "one RMSE win, one directional win, three more cells with promising-but-unproven point estimates."

This is not a downgrade of the project — it's a more credible version of the story to tell K2's traders.

---

## Slide 9 — Step #3: Per-regime methodology

The L3 model is "regime-switching": it assumes the market alternates between, say, a **calm** state and a **volatile** state, and uses different coefficients in each. The model never tells us which state we're "really" in — it only gives probabilities.

**The question this slide sets up**: when L3 wins on a cell, *which regime* is the win coming from? The calm 90% of days, or the volatile 10%?

**The procedure**: pull the regime probabilities from the full-history fit, hard-label each day with its most-likely regime, then split the cross-validation residuals (errors) by regime label and compute per-regime performance metrics.

**Important caveat**: these regime labels come from a model trained on *all* the data — they're hindsight assignments. So this slide is a *diagnostic* tool ("which regime carries the win?"), not a deployable per-regime trading rule (we can't know in real time which regime we're in).

---

## Slide 10 — Per-regime slices for TTF h=1, POWER h=1, POWER h=3, POWER h=5

**TTF h=1 L3** (an L3 win): The CV window is 96.7% calm regime, 3.3% turbulent. In the calm regime, L3 beats naive by 2% with directional accuracy 55%. In the turbulent regime, L3 has DA 25% — a disaster, but on only 16 observations. So the cell wins overall, but it's almost entirely a calm-regime story. Notable detail: the FV (fixed-variance) configuration means the two regimes have *identical* volatility — they differ only in coefficients, which is a slightly suspicious pattern.

**POWER h=1 L3** (an L3 win): The CV window is 78% calm, 22% turbulent. The calm regime beats naive by 3.2% with DA 56.6%; the turbulent regime breaks even. **Healthy** SV (switching-variance) separation: calm σ=0.025, turbulent σ=0.072. Roughly 3× volatility ratio.

**POWER h=3 L3** (a cell V4.3 vetoed on RMSE but kept directionally): Three regimes. The calm 73% has DA 60%; the mid-vol 7% has DA 75% (!); the high-vol 20% has DA 36% (sign-inverted). **This is why V4.3 saw a strong PT signal but the standalone logistic classifier couldn't reproduce it**: the directional skill is *inside the regime structure*. A future "regime-gated classifier" could exploit this — predict directionally only when we're confident we're not in regime 2.

**POWER h=5 L3** (vetoed by V4.3): Three regimes with **identical variance** — degenerate FV. Every regime is RMSE-worse than naive. The veto was correct in every regime slice.

---

## Slide 11 — Per-regime slices for EUA h=3, EUA h=5

Both cells were RMSE-vetoed by V4.3, and the per-regime view shows why: in both cells, **every regime** has L3 RMSE worse than naive and DA below 50%. There's no "good regime" hiding inside the bad average — L3 just doesn't have anything to say about EUA returns at h=3/h=5 in any state. The Track 3 sign-classifier wins on EUA h=3 because it uses the features differently — not because L3 was being unfairly judged.

**The big-picture lesson** at the bottom of this slide: L3 wins are calm-regime wins. Three FV configs are degenerate (identical regime variances) and should be re-checked under forced-SV. The POWER h=3 regime-bimodal pattern is the most interesting unfinished story in the whole deck.

---

## Slide 12 — Step #4: Rolling-window methodology

V4.3's cross-validation uses an **expanding window**: each fold trains on rows 0 through `t`, tests on rows `t` through `t+block_size`. So late folds have seen ALL the early data plus everything since.

This is fine for a backtest, but it doesn't tell us whether the model survives in a deployment context where data drifts and you can't rely on having a long history of relevant training data.

**Rolling-window CV** uses the same test windows but caps the training set at a fixed length (504 days ≈ 2 trading years). Same model spec, same hyperparameters — only the training slice changes.

**Decision rule**:

- If expanding ≈ rolling → edge is robust to training-history length.
- If rolling collapses → the win was partly an artefact of deep history.
- If rolling improves → maybe the early data is hurting the fit (regime shift, data quality issue).

---

## Slide 13 — Rolling vs expanding RMSE

A big table comparing each cell's RMSE under expanding vs rolling.

**The pattern is striking and clean**:

- **Every L1 cell is invariant** (RMSE delta < 0.1%). L1 with α=10,000 has thrown most signal away anyway, so training-set length barely matters.
- **Every policy-selected L3 cell is invariant** (TTF h=1, POWER h=1) — the wins survive the rolling test perfectly.
- **Every policy-vetoed L3 cell with deep regime structure collapses 10–18% further** under rolling (TTF h=5, POWER h=3, POWER h=5 all show big rolling-vs-expanding gaps).

**Translation**: V4.3's policy didn't just veto noise — it correctly identified which cells were brittle. That's strong validation of the policy layer itself.

---

## Slide 14 — Rolling vs expanding for sign classifier

Sign-classifier DA changes by ≤2pp in either direction across every cell when training is capped to 504 days. The flagship **EUA h=3 win** ticks up slightly (55.3 → 55.5). EUA h=1 actually *improves* from 48.8 to 50.7 — possibly because early data was noise.

The reason these are so flat: logistic regression with C=0.01 (heavy regularisation) is slow to change as you give it more data. Stable across protocols, but at the cost of small absolute DA improvements.

---

## Slide 15 — Final disclosure summary table

A consolidated 6-column matrix: cell × (V4.3 claim, bootstrap CI, rolling delta, per-regime view, Track 4 verdict).

The verdict column is the Track 4-validated truth:

- **TTF h=1 L3**: Robust ✓
- **EUA h=3 sign**: Robust ✓
- **POWER h=1 L3**: Promising but bootstrap CI just-crosses zero
- **EUA h=5 sign**: Promising, CI just-crosses
- **TTF h=1 PT-claim** (the V4.2 directional claim): magnitude OK, sign skill drifted
- **POWER h=3 L3**: regime-gated classifier is the natural follow-up
- **POWER h=5 L3** (V4.2's "win"): veto correct on every test

Three big-number cards at the bottom:

- **Bootstrap-clean wins: 2/9**
- **Promising-but-unproven: 3/9**
- **Veto correctness rate: 100%** ← every cell V4.3 routed to naive showed worse rolling-CV behaviour AND bootstrap-confirmed underperformance.

That last number is the single most important number on the slide. It says the policy layer V4.3 introduced is *demonstrably* doing the right thing on this dataset — not just on the metric it was tuned to optimise.

---

## Slide 16 — Open items / next steps

Four follow-ups in priority order:

1. **POWER h=3 regime-gated classifier** — biggest expected new deliverable. Per-regime slice gave us the recipe.
2. **Forced-SV sensitivity** for the three degenerate FV configs (TTF h=1, POWER h=5, EUA h=5). Quick rerun.
3. **BBG vol-surface re-pull** — the original V4.4 attempt was killed by data-side issues, not modeling-side. If IT fixes the 2021–23 hole we can revisit.
4. **Refresh `k2_clean` to cover 2025–2026** — current frozen CSV stops at 2024-12-30. Bringing it forward 16 months is the cheapest single thing that improves every CI in this deck.

---

## Slide 17 — Bottom line

**What we can stand behind**:

- TTF h=1 L3 — the magnitude flagship is real (DM, bootstrap CI, rolling-window all clear).
- EUA h=3 sign — the directional flagship is real (DA, PT, EV, Sharpe CIs all clear).
- V4.3's policy gate — every veto is independently re-validated. **100% veto correctness**.

**What needs caveats**:

- POWER h=1 L3 has the strongest point estimate in the deck but bootstrap CI is borderline. Encouraging, not deployable.
- EUA h=5 sign-PT is on the edge of significance after bootstrap.
- POWER h=3's regime-bound directional edge is real but needs a regime-gated classifier to deploy.

**What changed about V4.3's story**:

- V4.3 said: "two RMSE wins, two directional wins."
- Track 4 says: "one RMSE win, one directional win, three promising-but-unproven cells, **and a policy gate that earns trust by correctly rejecting everything else**."

> **The headline**: Track 4 makes V4.3 deliverable to a real trading desk. Not "more wins" — fewer, better-supported wins, with a transparent audit trail showing why every other cell is honestly routed to naive. That's the difference between a paper-promising model and one a desk can actually use.

---

## Appendix A — Feature inventory by target

This is the full candidate feature set that L1 (Ridge) sees, per market. Every feature is lag-1 unless noted (`_lagN` = that specific lag of the return). Source: `features_TTF / features_POWER / features_EUA` in `K2E_Model_V4_2.py`. V4.3 and V4.4 do not modify these blocks.

### TTF (Dutch gas, target `LR_TTF`) — 43 candidate features

- **Own AR lags (4)**: `LR_TTF_lag1/2/3/5`
- **Cross-market lags (5)**: `LR_POWER_lag1`, `LR_TTF_DA_lag1`, `LR_COAL_lag1`, `LR_BRENT_lag1`, `LR_JKM_lag1`
- **Fundamentals (7)**: `D_GAS_CONS_ANOM`, `TEMP_ANOM_WARM`, `TEMP_ANOM_COOL`, `STORAGE_DEV`, `LR_LNG_FLOW`, `LR_NOR_GAS`, `D_RUS_PIPE`
- **Storage buffer (2)**: `NO_BUFFER_M1`, `LOG_BUFFER_M1`
- **Realised vol (2)**: `RVOL20_TTF`, `RVOL5_TTF`
- **Interactions (2)**: `LR_JKM_x_NOBUF`, `LR_COAL_x_NOBUF`
- **COT positioning (2)**: `D_COT_TTF_HF_NET`, `COT_TTF_HF_RATIO`
- **V4 derived — term structure & options (5)**: `TTF_TS`, `D_TTF_TS`, `LR_TTF_CAL`, `LR_TTF_IV`, `D_TTF_PREM`
- **V4.1 GIE EU storage & flows (5)**: `GIE_EU_FULL_DOY_DEV`, `GIE_EU_NET_WD`, `D_GIE_EU_NET_WD`, `GIE_EU_INJ_LR`, `GIE_EU_WD_LR`
- **V4.1 DE outages (2)**: `GIE_DE_OUT_UNPL`, `GIE_DE_OUT_TOT_LOG1P`
- **V4.2 BBG (7)**: `BBG_LR_TTF_M1_M2`, `D_BBG_LR_TTF_M1_M2`, `BBG_LR_TTF_M1_M3`, `BBG_LR_NBP_M1_M2`, `D_LOG_OI_TZTA`, `D_LOG_VOL_TZTA`, `BBG_LR_BDIY`

### POWER (German baseload, target `LR_POWER`) — 32 candidate features (+1 optional)

- **Own AR lags (4)**: `LR_POWER_lag1/2/3/5`
- **Cross-market lags (5)**: `LR_TTF_lag1`, `LR_EUA_pos`, `LR_EUA_neg`, `LR_COAL_lag1`
- **Demand / weather (5)**: `LR_LOAD`, `TEMP_ANOM_WARM`, `TEMP_ANOM_COOL`, `WATER_ANOM_SCAN`, `WATER_ANOM_ALP`
- **Renewables / nuclear supply (3)**: `LR_WIND`, `LR_SOLAR`, `LR_NUCLEAR_FR`
- **Storage buffer (1)**: `NO_BUFFER_M1`
- **Realised vol (2)**: `RVOL20_POWER`, `RVOL5_POWER`
- **Interactions (2)**: `LR_TTF_x_NOBUF`, `LR_EUA_x_NOBUF`
- **V4 derived — spark & term structure (5)**: `SPARK_Z20`, `D_CLEAN_SPARK`, `POWER_TS`, `D_POWER_TS`, `LR_POWER_CAL`
- **V4.1 GIE DE storage & outages (5)**: `GIE_DE_FULL_DOY_DEV`, `GIE_DE_NET_WD`, `D_GIE_DE_NET_WD`, `GIE_DE_OUT_UNPL`, `GIE_DE_OUT_TOT_LOG1P`
- **V4.2 BBG (3)**: `BBG_LR_POWER_BASE_PEAK`, `D_BBG_PX_RTEGNUCD` (French nuclear shock), `BBG_LR_COAL_XA1`
- **Conditional (+1)**: `LR_RESID_LOAD` — added only if present in the clean dataset.

### EUA (EU carbon, target `LR_EUA`) — 37 candidate features

- **Own AR lags (4)**: `LR_EUA_lag1/2/3/5`
- **Cross-market lags (6)**: `LR_TTF_lag1`, `LR_POWER_lag1`, `LR_COAL_lag1`, `LR_COAL_GS`, `LR_COAL_GS_CAL`, `LR_BRENT_lag1`
- **Demand / macro (2)**: `TEMP_ANOM_COOL`, `LR_LOAD`
- **Gas-supply fuel-switch channel (2)**: `D_RUS_PIPE`, `LR_LNG_FLOW`
- **Storage buffer (3)**: `NO_BUFFER_M1`, `NO_BUFFER_CAL`, `LOG_BUFFER_CAL`
- **Realised vol (3)**: `RVOL20_EUA`, `RVOL5_EUA`, `RVOL20_TTF`
- **Interactions (1)**: `LR_COAL_x_NOBUF`
- **COT — HF cohort (2)**: `D_COT_EUA_HF_NET`, `COT_EUA_HF_RATIO`
- **V4 COT — compliance cohort (3)**: `D_COT_EUA_COMPL_NET`, `COT_EUA_COMPL_vs_HF`, `D_COT_EUA_COMPL_vs_HF`
- **V4 derived — spark & gas TS (2)**: `SPARK_Z20`, `TTF_TS`
- **V4.1 GIE (3)**: `GIE_EU_FULL_DOY_DEV`, `GIE_EU_NET_WD`, `GIE_DE_OUT_TOT_LOG1P`
- **V4.2 BBG — term structure, IV, auction, OI (7)**: `BBG_LR_EUA_M1_M2`, `D_BBG_LR_EUA_M1_M2`, `BBG_IVOL_MO1`, `D_BBG_IVOL_MO1`, `BBG_IVOL_MO1_Z20`, `BBG_LR_EUA_AUCTION_VS_MO1`, `D_LOG_OI_MO1`

### Shared design choices

- Target `y` is the `h`-day cumulative log-return; targets are `LR_TTF / LR_POWER / LR_EUA` at horizons h=1, 3, 5.
- Every feature is lagged by 1 day (via `L(·, 1)`) to prevent look-ahead — except the three non-lagged storage-state dummies (`NO_BUFFER_M1`, `NO_BUFFER_CAL`).
- L1 (Ridge, tuned α — ends up at 10,000 on 8/9 cells, 1,000 for EUA h=5) sees the full block.
- L3 (Markov-switching) is handed the top-`k` L1-ranked features per cell, with the AR-1 lag force-included (`_select_top_features` in `K2E_Model_V4_2.py`).

---

## Appendix B — Features actually selected by each cell's L3 winner

These are the features that ended up inside the final regime-switching model for each cell. They come from L1's absolute-coefficient ranking, with the AR-1 lag force-included, and capped at `k_feats` — the value V4.3's L3 sweep chose. Ranking/values below are reproduced by running the V4.2/V4.3 pipeline on `k2_clean_v4.csv`.

Cells vetoed by V4.3's policy gate (routed to naive) do not have a "final L3 model" in the deliverable sense, so for those cells we show the top-7 L1 ranking for diagnostic reference only.

### Bootstrap-clean wins (the two cells K2 can stand behind)

**TTF h=1 L3** — config `k=3, s=2, FV`; α=10,000
- **Selected (in order):** `LR_TTF_lag1` (AR-forced), `D_RUS_PIPE`, `LR_COAL_lag1`
- **Reading:** the magnitude flagship runs on own-lag momentum + one geopolitical flow shock (Russian pipeline) + one fuel-linkage control (coal). Deliberately tiny.

**EUA h=3 sign classifier** (logistic, not L3) — uses the full L1 feature block with C=0.01 regularisation, no top-k truncation. The features doing the most work (by |coef|):
- `BBG_IVOL_MO1`, `LR_EUA_lag3`, `COT_EUA_COMPL_vs_HF`, `BBG_IVOL_MO1_Z20`, `D_RUS_PIPE`, `RVOL20_EUA`, `LR_EUA_lag5`
- **Reading:** implied-vol level + compliance-vs-HF positioning + own-return memory is what separates up-days from down-days at the 3-day horizon.

### Promising but unproven (bootstrap CI just-crosses zero)

**POWER h=1 L3** — config `k=5, s=2, SV`; α=10,000
- **Selected:** `LR_POWER_lag1` (AR-forced), `LR_COAL_lag1`, `LR_POWER_lag5`, `LR_EUA_pos`, `WATER_ANOM_SCAN`
- **Reading:** own-lag momentum, coal input cost, asymmetric EUA linkage (only positive EUA moves transmit), and Scandinavian hydro water anomaly. Healthy SV separation (σ_calm=0.025 vs σ_turb=0.072) — see slide 10.

**POWER h=3 L3** — config `k=3, s=3, SV`; α=10,000 — *RMSE-vetoed but directionally interesting*
- **Selected:** `LR_POWER_lag1` (AR-forced), `GIE_DE_NET_WD`, `LR_POWER_lag2`
- **Reading:** German storage net-withdrawal is the single non-autoregressive feature. The per-regime slice (slide 10) showed states 0 and 1 give DA 60%/75% and state 2 is sign-inverted — the recipe for a regime-gated directional classifier is "short list of features, three regimes, gate on regime ≠ 2."

### V4.3-vetoed cells (top-7 L1 ranking only — no L3 model delivered)

**TTF h=3** — top L1 features: `D_RUS_PIPE`, `LOG_BUFFER_M1`, `BBG_LR_TTF_M1_M2`, `GIE_EU_NET_WD`, `TEMP_ANOM_COOL`, `LR_TTF_lag2`, `TEMP_ANOM_WARM`
**TTF h=5** — top L1: `LOG_BUFFER_M1`, `D_RUS_PIPE`, `TEMP_ANOM_COOL`, `BBG_LR_TTF_M1_M2`, `GIE_EU_NET_WD`, `GIE_DE_OUT_UNPL`, `TEMP_ANOM_WARM`
**POWER h=5** — config `k=3, s=3, FV` (degenerate variance); selected: `LR_POWER_lag1` (AR-forced), `GIE_DE_NET_WD`, `GIE_DE_OUT_TOT_LOG1P`. Veto correct on every test.
**EUA h=1** — top L1: `D_RUS_PIPE`, `LR_BRENT_lag1`, `LR_COAL_lag1`, `BBG_IVOL_MO1`, `COT_EUA_COMPL_vs_HF`, `RVOL20_EUA`, `LR_EUA_lag5`
**EUA h=3** — top L1: `BBG_IVOL_MO1`, `LR_EUA_lag3`, `COT_EUA_COMPL_vs_HF`, `BBG_IVOL_MO1_Z20`, `D_RUS_PIPE`, `RVOL20_EUA`, `LR_EUA_lag5`
**EUA h=5** — top L1 (α=1,000 here, not 10,000): `BBG_IVOL_MO1`, `LR_EUA_lag3`, `BBG_IVOL_MO1_Z20`, `LR_BRENT_lag1`, `TEMP_ANOM_COOL`, `RVOL20_EUA`, `D_RUS_PIPE`

### Patterns worth noticing across the selections

- **`D_RUS_PIPE` is the single most-picked feature** across targets — it's top-7 in TTF at every horizon, EUA at every horizon, and is one of three features in the TTF h=1 flagship. Geopolitical gas-flow shocks transmit to both gas and carbon pricing through the fuel-switch channel.
- **`BBG_IVOL_MO1` dominates EUA** at every horizon — option-implied vol is the strongest single EUA signal and is what the sign classifier mainly rides.
- **`GIE_DE_NET_WD`** is picked by every POWER L3 cell — German gas storage net-withdrawal is effectively the German power market's input-cost pulse.
- **Own-lag dominance is horizon-dependent.** At h=1 the AR-1 lag is a large-coefficient feature naturally; at h=3/5 the AR-forced lag only enters because the selection rule inserts it — L1 would have ranked it below fundamentals.
- **EUA cells share nearly the same top features** across horizons — the difference between the sign-classifier win at h=3 and the veto at h=5 is not feature content, it's noise level relative to the usable signal.
