# Key to Energy (K2E) — European Energy Futures Forecasting

Fordham University × Key to Energy collaboration. Daily forecasting models for
European energy futures — **TTF gas (M+1)**, **German power (M+1)**, and **EUA
carbon (Cal+1)** — at the D+1, D+3, and D+5 horizons, trained on 2021–2024 and
evaluated out-of-sample through mid-2026.

## Repository structure

```
code/
  model/              V4.x forecasting pipeline
    K2E_Model_V4_2.py   feature engineering + Layer-1 (Ridge-ARX) + Layer-3 (Markov-switching ARX)
    K2E_Model_V4_3.py   best-of-cell policy gate + directional sign classifier
    K2E_Model_V4_4.py   deployed wrapper + Track-4 diagnostics
    K2E_Bootstrap.py    bootstrap confidence intervals
    K2E_RegimeDiag.py   per-regime diagnostics
    K2E_RollingCV.py    rolling-window CV stability check
  data_pipeline/      data assembly (not needed to reproduce the eval)
    bbg_merge_v2.py     Bloomberg extract -> k2_clean merge
    gie_fetcher.py      GIE AGSI storage/outage API pull
    gie_merge.py        merge GIE series into k2_clean
  full_oos_eval/      out-of-sample evaluation (this round of work)
    parse_full.py            raw K2E monthly Excel -> model schema (positional, validated)
    predict_oos.py           deployed model -> OOS forecasts (all 9 cells + EUA signal)
    eval_oos.py              MAE / RMSE / R²-vs-naive / directional accuracy by period
    predict_all_cells.py     naive vs L1 vs L3 on all 9 cells, ignoring the policy gate
    walk_forward_directional.py  rolling-refit directional backtest
    war_split_eval.py        pre/post Israel–Iran-war (2025-06-13) regime split
docs/
  k2_clean_v4_data_dictionary.md
RESULTS.md            full out-of-sample findings (Jan 2025 – Jun 2026)
```

## Data

**No data is committed to this repository.** The inputs are proprietary
(Bloomberg terminal extracts and Key to Energy monthly deliveries) and must not
be redistributed. Place the required files in a local `data/` directory
(git-ignored) to run the pipeline:

- `data/k2_clean_v4.csv` — merged training panel, 2021–2024
- `data/full_parsed.csv` — full dataset parsed to model schema (output of `parse_full.py`)
- `data/oos_jan2025_parsed.csv` — column-schema reference for the parser

GIE AGSI storage/outage series are publicly available and reproducible via
`code/data_pipeline/gie_fetcher.py`.

## Reproduce the out-of-sample evaluation

```bash
python code/full_oos_eval/parse_full.py "data/Full Dataset.xlsx" data/full_parsed.csv
python code/full_oos_eval/predict_oos.py            # -> data/full_oos_predictions.csv
python code/full_oos_eval/eval_oos.py               # -> metrics by cell × period
python code/full_oos_eval/predict_all_cells.py      # naive vs L1 vs L3, all cells
python code/full_oos_eval/walk_forward_directional.py
python code/full_oos_eval/war_split_eval.py
```

## Headline finding

Over the full ~18-month out-of-sample window the model does **not** beat a naive
random walk at the daily horizon on any cell, and no directional signal survives
a walk-forward backtest — consistent with near-efficient pricing of liquid
energy futures. The earlier 22-day (Jan-2025) highlights were small-sample
noise. See **[RESULTS.md](RESULTS.md)** for the full analysis, including the
all-layer comparison and the pre/post-war regime split.
