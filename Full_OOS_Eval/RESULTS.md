# K2E V4.4 — Full Out-of-Sample Evaluation (Jan 2025 → Jun 2026)

**Date:** 2026-06-26
**Input:** `Full Dataset.xlsx` (2021-01-05 → 2026-06-11, K2E monthly-delivery format)
**Model:** V4.4 deployed model, trained on 2021-01 → 2024-12 (frozen, no refit)
**OOS window:** 367 trading days (≈18 months) — vs. the original deck's 22 days (Jan-2025 only)

---

## TL;DR — How predictive is the model?

**At the daily price-forecast task, the model does not beat a naive random walk out of sample, and the directional signals did not hold up.** The encouraging cells in the original 22-day (Jan-2025) deck — TTF D+5 at 77% directional accuracy and the EUA D+3 signal at ~74% — were small-sample artifacts that vanish over 18 months. This is the expected behaviour for liquid, near-efficient energy futures at a daily horizon.

The exercise was still worth doing: it converts an optimistic 1-month read into a statistically meaningful 18-month verdict, and the pipeline now runs end-to-end on any future K2E delivery.

---

## Price-forecast metrics — full 18-month OOS

| Market | Horizon | N | MAE vs naive | R² vs naive | Dir. acc. | Model used |
|--------|---------|---|-------------|-------------|-----------|------------|
| TTF | D+1 | 366 | **−5.2%** | −0.07 | 50.0% | L3 Markov |
| TTF | D+3 | 364 | 0.0% | 0.00 | — | naive |
| TTF | D+5 | 362 | **−62.2%** | −1.33 | 50.0% | L3 Markov |
| POWER | D+1 | 366 | **−4.0%** | −0.07 | 52.6% | L3 Markov |
| POWER | D+3 | 364 | 0.0% | 0.00 | — | naive |
| POWER | D+5 | 362 | 0.0% | 0.00 | — | naive |
| EUA | D+1/D+3/D+5 | — | 0.0% | 0.00 | — | naive |

*Negative MAE-vs-naive / negative R² = the model is **worse** than just carrying today's price forward.*
*"naive" cells are tied to the random walk by construction (the deployed policy selected naive for them on the 2021-24 CV), so they score 0% by definition.*

**Read:** The three cells where the deployed policy actually runs a model (TTF D+1, TTF D+5, POWER D+1, all L3 Markov-switching) all land at or slightly **below** the random walk on error, and at a coin-flip (50–53%) on direction. No price-level edge survives out of sample.

## The Jan-2025 "wins" were noise

| Cell | Jan-2025 (22d) | Full 2025 (255d) | Full OOS (367d) |
|------|---------------|------------------|-----------------|
| TTF D+5 directional acc. | **77.3%** | 49.8% | 50.0% |
| POWER D+1 MAE vs naive | +2.4% | −0.3% | −4.0% |
| EUA D+3 directional acc. | ~74% (deck) → 31.8% (rebuilt) | 45.1% | 46.7% |

Every cell that looked good on 22 days regressed to ~50% (or worse) once the sample grew.

## EUA D+3 directional signal

| Period | Hit rate | Notes |
|--------|----------|-------|
| Training walk-forward CV | 55.3% (PT p=0.008) | Significant **in sample** |
| Full OOS (18 mo) | **46.7%** | Below coin-flip; baseline drifts |

The logistic sign classifier has a genuine in-sample directional edge (55%, p<0.01) but it does not generalise: out of sample the predicted-UP rate collapses to ~9% while EUA actually rose 54% of the time. The signal's class baseline is unstable to the input regime — a classic in-sample/out-of-sample breakdown.

---

## Method notes & caveats

- **Parser** (`code/parse_full.py`) maps the raw 101-column delivery to the model schema by position; validated against the original `oos_jan2025_parsed.csv` — **all 70 non-COT columns are bit-identical**. The only differences are in the weekly COT block, where the full dataset correctly carries December positioning into early January (the isolated single-month file could not).
- **Prediction** (`code/predict_oos.py`) reproduces the deployed model: **POWER D+1 matches the original to 1e-8**; TTF D+1/D+5 match at correlation > 0.98 (residual differences are EM-fit randomness in the Markov estimation plus the cleaner COT inputs). EUA price cells are naive ⇒ exact.
- The model is **frozen at end-2024** — this is a true out-of-sample test, not a refit. Refitting on a rolling window is the natural next experiment if directional trading is the goal.
- These artifacts (`data/`, `code/`) regenerate the full table via `python code/eval_oos.py`.

---

## Addendum — all 3 layers on all 9 cells (ignoring the policy gate)

To test whether the naive fallback was the right call, I re-fit **L1 (Ridge-ARX)** and **L3 (Markov-switching)** on **all 9 cells** — including the 6 where the deployed policy had discarded them — and ran them over the full 18 months. (`code/predict_all_cells.py`, `data/metrics_all_layers_oos.csv`.)

**Verdict: the policy gate was vindicated. No discarded model beats naive by a meaningful margin out of sample.**

| Cell | naive R² | **L1** R² (MAE vs naive) | **L3** R² (MAE vs naive) | OOS-best |
|------|---------|--------------------------|--------------------------|----------|
| TTF D+1 | 0.000 | +0.005 (+0.1%) | −0.071 (−5.2%) | L1 (+0.3%) |
| TTF D+3 | 0.000 | +0.002 (+0.3%) | −0.035 (−3.1%) | L1 (+0.1%) |
| TTF D+5 | 0.000 | +0.007 (+0.4%) | −0.497 (−25.8%) | L1 (+0.3%) |
| POWER D+1 | 0.000 | +0.001 (+0.0%) | *(EM fit failed this run)* | L1 (+0.1%) |
| POWER D+3 | 0.000 | +0.000 (−0.6%) | −0.824 (−45.3%) | ~tie |
| POWER D+5 | 0.000 | +0.003 (−0.5%) | −0.606 (−29.6%) | L1 (+0.2%) |
| EUA D+1 | 0.000 | +0.001 (−0.0%) | −0.006 (−0.6%) | L1 (+0.1%) |
| EUA D+3 | 0.000 | +0.002 (−0.5%) | −0.921 (−38.4%) | L1 (+0.1%) |
| EUA D+5 | 0.000 | −0.096 (−8.7%) | −1.650 (−58.6%) | **naive** |

Two clean takeaways:

1. **L1 ≈ naive.** In every cell the cross-validation drove the Ridge penalty to the top of the grid (α = 1e3–1e4), which shrinks the coefficients almost to zero — so L1's forecast collapses onto the random walk. It "wins" 8/9 cells but by ≤0.4% RMSE, i.e. statistically nothing. The model is telling us there is no exploitable daily price signal in these features.
2. **L3 is worse than naive everywhere** on RMSE (R² from −0.04 to −1.65). The Markov-switching layer — the one the original deck showcased — consistently *adds* error out of sample. It overfits the regime structure.

**One nuance worth flagging:** although L3's *magnitude* calibration is poor, its *directional* hit-rate is mildly above 50% in a few cells (POWER D+5 57.3%, EUA D+5 56.9%, POWER D+3 54.4%). So if the objective were a long/short directional signal rather than price RMSE, L3 is the only thread with any pull — but it's weak, inconsistent across cells, and the dedicated EUA D+3 directional classifier already failed out of sample (46.7%). I would not trade on it without a rolling refit and proper cost/Sharpe analysis.

*(Caveat: the Markov EM has random restarts, so L3 configs/values shift slightly run-to-run; POWER D+1's L3 final fit happened to fail this run. None of this changes the conclusion — L3 is reliably below naive on error.)*

---

## Addendum 2 — rolling-refit walk-forward directional backtest

The fairest test for a *directional* (long/short) signal is a walk-forward: at each OOS day, refit the logistic sign-classifier on all realised history (origin + h ≤ today, so no lookahead), re-tune the L2 penalty monthly, predict the next move, and roll on. Features restricted to those actually populated in the live monthly delivery (auto-dropped BBG/GIE, ffilled-options, and training-only derived columns by ≥95% OOS-coverage). (`code/walk_forward_directional.py`, `data/walkforward_directional_*.csv`.)

| Cell | N | Dir. acc. | PT p-value | EV (gross) | Sharpe* | pred-UP |
|------|---|-----------|-----------|-----------|---------|---------|
| TTF D+1 | 364 | 48.9% | 0.696 | +9.7 bps | +0.38 | 79% |
| TTF D+3 | 362 | 49.4% | 0.252 | +17.8 bps | +0.23 | 81% |
| **TTF D+5** | 360 | 52.5% | **0.048** | +72.7 bps | +0.57 | 84% |
| POWER D+1 | 363 | 51.8% | 0.237 | −12.2 bps | −0.54 | 52% |
| POWER D+3 | 364 | 46.7% | 0.953 | −61.2 bps | −0.95 | 61% |
| POWER D+5 | 361 | 48.2% | 0.973 | −109.8 bps | −1.04 | 70% |
| EUA D+1 | 358 | 50.8% | 0.317 | +8.5 bps | +0.77 | 80% |
| EUA D+3 | 356 | 53.1% | 0.493 | −6.6 bps | −0.21 | 84% |
| EUA D+5 | 355 | 46.8% | 0.961 | −50.8 bps | −0.97 | 62% |

**Verdict: no deployable directional edge survives the walk-forward.**

- Hit rates sit at **47–53% — coin-flip** across all 9 cells. Several cells lose money gross (POWER D+3/D+5, EUA D+5).
- Only **TTF D+5** clears PT p<0.10 (p=0.048). But that is **one significant cell out of nine tested** — with 9 independent tests you expect ~0.9 false positives at the 0.10 level, so a single p=0.048 (Bonferroni-adjusted ≈ 0.43) is what noise looks like. Its Sharpe* of +0.57 is also *before* transaction costs and is inflated by the overlapping 5-day windows (effective N ≈ 360/5 ≈ 72).
- The classifiers carry a persistent long bias (pred-UP 79–84% in TTF/EUA), so they mostly ride the market's drift rather than time it.

Continuous refitting does **not** rescue the signal — the in-sample directional edge (55%, PT p=0.008) was overfitting, not a stable, tradeable phenomenon.

*Sharpe\* is a crude annualised gross figure that ignores transaction costs, the h-day return overlap/autocorrelation, and rebalancing — read it as directional, not as a backtest P&L.*

---

## Addendum 3 — pre-war vs post-war (Israel–Iran war, 13 Jun 2025)

Hypothesis: the model may have had genuine predictive power in the calmer pre-war regime, with the June-2025 oil/gas shock destroying the aggregate OOS metrics. Each forecast is bucketed by its full h-day window: **PRE** (window ends before 13 Jun 2025), **STRADDLE** (window crosses the shock), **POST**. (`code/war_split_eval.py`, `data/war_split_directional.csv`. The cutoff is a parameter — easy to change.)

**Hypothesis not supported — the pre-war regime is, if anything, slightly worse.**

- **Mean directional accuracy across the 9 cells: PRE = 47.8%, POST = 50.7%.** The pre-war "calm" period is *below* coin-flip and below the post-war period. There is no hidden edge that the war erased.

Deployed price models, pre-war:

| Cell | N | MAE vs naive | R² vs naive | Dir. acc. |
|------|---|-------------|-------------|-----------|
| TTF D+1 | 112 | −4.2% | −0.06 | 52.7% |
| TTF D+5 | 108 | −61.3% | −1.77 | 53.7% |
| POWER D+1 | 112 | +0.7% | −0.03 | **60.7%** |

The **one flicker** is **POWER D+1 direction** in the pre-war window: 60.7% (deployed) / 58.9% (walk-forward), PT p=0.057. But:

1. It's **one cell out of nine** — at p<0.10 you expect ~0.9 chance hits, so an isolated p=0.057 is not evidence.
2. It **reverses after the war**: walk-forward POWER D+1 POST drops to 48.8% (PT p=0.67, Sharpe −1.10), and the predicted-UP rate flips from 38% → 59%. A stable signal does not invert.

**The split also debunks the one full-sample "significant" cell.** TTF D+5's p=0.048 over the whole OOS turns out to be a *post-war* phenomenon: PRE 50.0% (p=0.55) vs POST 53.0% (p=0.014), with a lucky +604 bps straddle window. It's a persistently long-biased classifier (84% UP) riding the post-war price spike — not pre-war timing skill.

### On adding a regime adjustment
A regime dummy / regime-conditional intercept helps when a model forecasts *well within* a regime but the level or volatility shifts *across* regimes. Here the model is **not predictive within either regime** (PRE 47.8%, POST 50.7%), so a regime adjustment has nothing to amplify — it would only re-center an already-random signal. The single place it could matter is POWER D+1, whose directional bias flipped across the shock; a regime-conditional intercept would stop the inversion, but with PRE/POST samples this small that is curve-fitting the war, not discovering alpha. I would not pursue it as a source of edge.

## Files
- `data/full_parsed.csv` — full dataset parsed to model schema (1387 rows)
- `data/full_oos_predictions.csv` — model forecasts, all 9 cells + EUA signal (367 OOS rows)
- `data/metrics_full_oos.csv` — deployed-model metrics by cell × period (Full / Jan-2025 / 2025 / 2026-YTD)
- `data/metrics_all_layers_oos.csv` — naive vs L1 vs L3 on all 9 cells, full OOS
- `data/walkforward_directional_summary.csv` — rolling-refit directional backtest, 9 cells
- `data/walkforward_directional_detail.csv` — per-day signal, prob, realised return
- `data/war_split_directional.csv` — directional metrics by pre/straddle/post-war regime
- `code/` — parser, prediction pipeline, eval, all-layer comparison, walk-forward backtest, war-split eval
