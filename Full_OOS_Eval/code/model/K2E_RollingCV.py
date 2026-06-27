"""
Track 4 #4 — Rolling-window CV stability check.

V4.2's CV is *expanding*: each fold trains on rows[0:ts], tests on
rows[ts:te]. The training set grows fold-by-fold, so late-fold models have
seen the early data plus everything since. This is honest walk-forward, but
it doesn't tell us whether the win at e.g. TTF h=1 L3 survives if the model
only sees the most recent N days.

Rolling-window CV uses the same test windows but caps the training set at a
fixed length (default 504 ≈ two trading years). If a cell's metrics hold up
under both schemes, the edge is robust to training-history length. If
rolling collapses while expanding wins, the edge is partly an artefact of
deep history that the live system wouldn't have at deployment time.

Same model spec is used in both schemes (α from V4.3's L1, best (k,s,sv) from
V4.3's L3 sweep, C from V4.3's logit) — we vary ONLY the train slice.

Output: side-by-side table of expanding vs rolling RMSE / DA per cell.
"""
from __future__ import annotations

import numpy as np
import warnings
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error

from K2E_Model_V4_2 import (
    make_X_y, get_cv_folds, direction_accuracy,
    _select_top_features, _fit_markov, _smoothed_probs_array,
    _markov_predict,
)
from K2E_Model_V4_3 import _prepare_sign_train

warnings.filterwarnings('ignore')


def _rolling_folds(n: int, n_folds: int, train_window: int):
    """Yield (train_start, train_end, test_start, test_end) tuples.

    Test windows are identical to get_cv_folds(n, n_folds); training is
    truncated to the last `train_window` rows before each test window.
    """
    fold = n // (n_folds + 1)
    out = []
    for f in range(n_folds):
        ts = fold + f * fold
        te = ts + fold
        if te > n:
            break
        train_end = ts
        train_start = max(0, train_end - train_window)
        out.append((train_start, train_end, ts, te))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# L1 — Ridge-ARX with rolling window
# ─────────────────────────────────────────────────────────────────────────────

def rolling_cv_l1(X, y, alpha: float, train_window: int, n_folds: int = 5) -> dict:
    n = len(X)
    folds = _rolling_folds(n, n_folds, train_window)
    pred, true = [], []
    for trs, tre, ts, te in folds:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X.iloc[trs:tre])
        Xte = sc.transform(X.iloc[ts:te])
        ytr, yte = y.iloc[trs:tre], y.iloc[ts:te]
        p = Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xte)
        pred.extend(p)
        true.extend(yte.values)
    pred = np.asarray(pred); true = np.asarray(true)
    naive_rmse = float(np.sqrt(np.mean(true ** 2)))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    return {
        'rmse': rmse, 'da': direction_accuracy(true, pred),
        'naive_rmse': naive_rmse,
        'vs_naive_pct': (rmse - naive_rmse) / naive_rmse * 100.0 if naive_rmse > 0 else np.nan,
        'n': len(true),
    }


# ─────────────────────────────────────────────────────────────────────────────
# L3 — Markov-switching ARX with rolling window
# ─────────────────────────────────────────────────────────────────────────────

def rolling_cv_l3(X, y, l1_fit, market, k_feats, n_states,
                  switching_variance, train_window: int, n_folds: int = 3) -> dict:
    top_feats = _select_top_features(l1_fit, k_feats, market, force_ar=True)
    X_sub = X[top_feats]
    n = len(X_sub)
    folds = _rolling_folds(n, n_folds, train_window)
    pred, true = [], []
    n_converged = 0

    for trs, tre, ts, te in folds:
        Xtr = X_sub.iloc[trs:tre]
        ytr = y.iloc[trs:tre]
        Xte = X_sub.iloc[ts:te]
        yte = y.iloc[ts:te]

        sc = StandardScaler()
        Xtr_s = pd.DataFrame(sc.fit_transform(Xtr), columns=top_feats)
        Xte_s = pd.DataFrame(sc.transform(Xte), columns=top_feats)

        res = _fit_markov(ytr, Xtr_s, n_states, switching_variance)
        if res is None:
            p = Ridge(alpha=10.0).fit(Xtr_s.values, ytr.values).predict(Xte_s.values)
        else:
            n_converged += 1
            smooth = _smoothed_probs_array(res)
            last_probs = smooth[-1]
            p = _markov_predict(res, Xte_s, n_states, last_probs)
        pred.extend(p); true.extend(yte.values)

    pred = np.asarray(pred); true = np.asarray(true)
    naive_rmse = float(np.sqrt(np.mean(true ** 2)))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    return {
        'rmse': rmse, 'da': direction_accuracy(true, pred),
        'naive_rmse': naive_rmse,
        'vs_naive_pct': (rmse - naive_rmse) / naive_rmse * 100.0 if naive_rmse > 0 else np.nan,
        'n': len(true), 'n_converged': n_converged, 'n_folds': len(folds),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sign classifier — logistic with rolling window (re-uses C from V4.3 fit)
# ─────────────────────────────────────────────────────────────────────────────

def rolling_cv_sign(X, y, C: float, train_window: int, n_folds: int = 5) -> dict:
    n = len(X)
    folds = _rolling_folds(n, n_folds, train_window)
    sign_pred_all, y_all = [], []
    for trs, tre, ts, te in folds:
        Xtr_raw = X.iloc[trs:tre]; ytr_raw = y.iloc[trs:tre]
        Xtr, ytr_bin = _prepare_sign_train(Xtr_raw, ytr_raw)
        Xte, yte = X.iloc[ts:te], y.iloc[ts:te]
        if len(np.unique(ytr_bin)) < 2 or len(Xtr) < 20:
            prob = np.full(len(Xte), 0.5)
        else:
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
            mdl = LogisticRegression(penalty='l2', C=C, max_iter=500, solver='lbfgs')
            mdl.fit(Xtr_s, ytr_bin)
            prob = mdl.predict_proba(Xte_s)[:, 1]
        sp = np.where(prob > 0.5, 1.0, -1.0)
        sign_pred_all.extend(sp); y_all.extend(yte.values)
    sp = np.asarray(sign_pred_all); y_arr = np.asarray(y_all, dtype=float)
    mask = np.sign(y_arr) != 0
    da = float((np.sign(y_arr[mask]) == sp[mask]).mean() * 100.0) if mask.sum() > 0 else np.nan
    return {'da': da, 'n': int(mask.sum())}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def rolling_all(d, fitted: dict, results: list, sign_results: dict,
                train_window: int = 504) -> dict:
    """Run rolling-window CV for every cell already fitted by V4.3."""
    print(f"\n\n{'=' * 110}")
    print(f"TRACK 4 — ROLLING-WINDOW CV (train_window={train_window} days) vs EXPANDING (V4.3 baseline)")
    print('=' * 110)

    # Build a {(market,h,layer): expanding RMSE} lookup from V4.3 results
    exp_lookup = {(r['market'], r['h'], r['layer']): r for r in results}

    print(f"\n  RMSE / DA — L1 and L3 cells")
    print(f"  {'Market':6}{'h':>3} {'Layer':>5}   "
          f"{'Exp RMSE':>9}  {'Roll RMSE':>10}   "
          f"{'Δ%':>7}   {'Exp DA':>7}  {'Roll DA':>7}   {'note':<20}")
    print('  ' + '-' * 92)

    out = {}
    for market in ['TTF', 'POWER', 'EUA']:
        for h in [1, 3, 5]:
            X, y = make_X_y(d, market, h)
            # L1
            l1 = fitted.get((market, h, 'L1'))
            if l1 is not None:
                roll1 = rolling_cv_l1(X, y, alpha=l1['alpha'], train_window=train_window)
                exp1 = exp_lookup[(market, h, 'L1')]
                delta_pct = (roll1['rmse'] - exp1['cv_rmse']) / exp1['cv_rmse'] * 100.0
                out[(market, h, 'L1')] = roll1
                print(f"  {market:6}{h:>3} {'L1':>5}   "
                      f"{exp1['cv_rmse']:9.5f}  {roll1['rmse']:10.5f}   "
                      f"{delta_pct:+6.1f}%   {exp1['cv_da']:6.1f}%  {roll1['da']:6.1f}%   "
                      f"α={l1['alpha']:.4g}")
            # L3
            l3 = fitted.get((market, h, 'L3'))
            if l3 is not None:
                roll3 = rolling_cv_l3(X, y, l1_fit=l1, market=market,
                                      k_feats=l3['k_feats'], n_states=l3['n_states'],
                                      switching_variance=l3['switching_variance'],
                                      train_window=train_window)
                exp3 = exp_lookup.get((market, h, 'L3'))
                if exp3 is None:
                    continue
                delta_pct = (roll3['rmse'] - exp3['cv_rmse']) / exp3['cv_rmse'] * 100.0
                out[(market, h, 'L3')] = roll3
                sv_str = 'SV' if l3['switching_variance'] else 'FV'
                conv = f"{roll3['n_converged']}/{roll3['n_folds']}"
                print(f"  {market:6}{h:>3} {'L3':>5}   "
                      f"{exp3['cv_rmse']:9.5f}  {roll3['rmse']:10.5f}   "
                      f"{delta_pct:+6.1f}%   {exp3['cv_da']:6.1f}%  {roll3['da']:6.1f}%   "
                      f"k={l3['k_feats']} s={l3['n_states']} {sv_str} ({conv})")

    # Sign classifier
    print(f"\n  DA — Sign classifier (logit, C from V4.3)")
    print(f"  {'Market':6}{'h':>3}   {'Exp DA':>7}  {'Roll DA':>7}   "
          f"{'Δpp':>5}   {'C':>6}")
    print('  ' + '-' * 60)
    for (market, h), entry in sign_results.items():
        cv = entry['cv']
        ev = entry['metrics']
        X, y = make_X_y(d, market, h)
        roll = rolling_cv_sign(X, y, C=cv['C'], train_window=train_window)
        out[(market, h, 'SIGN')] = roll
        delta_pp = roll['da'] - ev['da']
        print(f"  {market:6}{h:>3}   {ev['da']:6.1f}%  {roll['da']:6.1f}%   "
              f"{delta_pp:+5.1f}   {cv['C']:>6.2f}")

    return out
