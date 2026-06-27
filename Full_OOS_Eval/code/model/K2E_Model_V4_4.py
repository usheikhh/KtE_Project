"""
K2 Energy / Fordham University Collaboration
Phase 1 v4.4: V4.3 pipeline, data-path refresh only (k2_clean_v4.csv).

What this is
------------
A thin wrapper over V4.3. The BBG ESMA COT and vol-surface additions from the
new extract were tested and did not improve on V4.3 — see rationale below.
This module exists so future work can build on the v4 data file without
re-touching V4.3.

What was tested and dropped
---------------------------
1. Vol surface (BBG Option sheet — RK910/RK903/RK908/RK901): structural
   2-year black hole in the BBG pull (MO1 empty 2021-2023, TZT1 empty
   2022-2023). Not a modeling issue — needs re-pull with
   HIST_PUT_IMP_VOL_25DELTA_DFLT / HIST_CALL_IMP_VOL_25DELTA_DFLT on generic
   front tickers. Single-point ATM IV (BBG_IVOL_MO1) already wired in V4.2.

2. BBG ESMA COT F+O features on EUA: 5 candidates (D_SPEC_NET_FO,
   SPEC_NET_Z52, SPEC_NET_PCTGROSS, SPEC_LONG_PCTOI_FO, D_HEDGE_NET_FO) and
   1-feature trim (SPEC_NET_PCTGROSS only) both regressed vs V4.3 on the EUA
   sign classifier:
     h=3 PT p 0.008 → 0.032, Sharpe +0.89 → +0.68
     h=5 PT p 0.036 → 0.033, Sharpe +0.54 → +0.47
   Diagnosis: collinear with the MF_COT_EUA_* market-feed block V4.3 already
   uses — no orthogonal signal.

Data plumbing is kept: bbg_merge_v2.py still produces k2_clean_v4.csv with the
BBG COT columns populated, so future experiments (interactions, regime-gated
use) can reference them without re-running the merge.
"""

import sys

import K2E_Model_V4_3 as v43
from K2E_Bootstrap import bootstrap_all
from K2E_RegimeDiag import regime_diag_all
from K2E_RollingCV import rolling_all


def train_all_v44(train_path, sweep_l3=True, bootstrap_B=1000, bootstrap_alpha=0.10,
                  rolling_window=504):
    """Run V4.3 pipeline, then Track 4 diagnostics:
       (#1) safe-fit wrapper [in V4.3], (#2) bootstrap CIs, (#3) per-regime L3 slice,
       (#4) rolling-window CV stability check.
    """
    out = v43.train_all_v43(train_path, sweep_l3=sweep_l3)
    fitted, d, results, sweep_log, preds, policy_rows, sign_results = out
    boot = bootstrap_all(preds, sign_results, B=bootstrap_B, alpha=bootstrap_alpha)
    regime = regime_diag_all(d, fitted, preds)
    rolling = rolling_all(d, fitted, results, sign_results, train_window=rolling_window)
    return fitted, d, results, sweep_log, preds, policy_rows, sign_results, boot, regime, rolling


if __name__ == '__main__':
    train_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--train' and i + 1 < len(sys.argv):
            train_path = sys.argv[i + 1]
    if train_path is None:
        train_path = ('/Users/jefflu/Library/CloudStorage/OneDrive-Personal/'
                      'Documents/MSQF Fordham/ML Internship/Workspace/'
                      'KtE_Project/Jeff workspace/k2_clean_v4.csv')
    out = train_all_v44(train_path, sweep_l3=True)
