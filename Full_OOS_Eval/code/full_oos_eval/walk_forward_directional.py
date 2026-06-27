import os
"""Walk-forward (expanding-window) directional backtest for all 9 cells.

At each OOS origin t:
  - train a logistic sign-classifier on ALL history whose target is already
    realised (origin + h <= t)  -> no lookahead
  - re-tune the L2 penalty C monthly (held between re-tunes)
  - predict sign of the h-day-ahead return at t
Records realised hit-rate, Pesaran-Timmermann p, and a crude hard-signal
long/short EV & Sharpe (gross of costs).

Features are restricted to those actually present in the K2E monthly delivery
(BBG_* and GIE_* dropped), so this is a deployable signal, not a backfit.
"""
import sys, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
from K2E_Model_V4_2 import (
    load_clean, engineer_features, FEAT_FNS, TARGET_LR, get_cv_folds, pt_test,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MARKETS = ['TTF', 'POWER', 'EUA']
HORIZONS = [1, 3, 5]
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
RETUNE_EVERY = 21          # trading days (~monthly)
TRAIN = 'data/k2_clean_v4.csv'


def deployable_cols(feats, oos_mask, min_cov=0.95):
    """Keep only features actually populated in the live OOS delivery
    (auto-drops BBG/GIE, ffilled-options, and training-only derived cols)."""
    sub = feats.loc[oos_mask]
    return [c for c in feats.columns if sub[c].notna().mean() >= min_cov]


def tune_C(X, y):
    """log-loss CV tune over expanding folds (drops ties in train)."""
    n = len(X); folds = get_cv_folds(n, 5)
    scores = {c: [] for c in C_GRID}
    for ts, te in folds:
        m = y.iloc[:ts].values != 0
        Xtr = X.iloc[:ts][m]; ytr = (np.sign(y.iloc[:ts].values[m]) > 0).astype(int)
        if len(np.unique(ytr)) < 2 or len(Xtr) < 30:
            continue
        sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(X.iloc[ts:te]); yte = (np.sign(y.iloc[ts:te].values) > 0).astype(int)
        for c in C_GRID:
            mdl = LogisticRegression(C=c, max_iter=500).fit(Xtr_s, ytr)
            p = mdl.predict_proba(Xte_s)[:, 1]
            eps = 1e-9
            scores[c].append(-(yte*np.log(p+eps)+(1-yte)*np.log(1-p+eps)).mean())
    means = {c: np.mean(s) if s else np.inf for c, s in scores.items()}
    return min(means, key=means.get) if np.isfinite(min(means.values())) else 0.01


# ---- build one continuous engineered dataset (train + full OOS) ----
df_train = load_clean(TRAIN)
full = pd.read_csv('data/full_parsed.csv', parse_dates=['Date'])
train_end = pd.to_datetime(df_train['Date']).max()
oos_raw = full[full['Date'] > train_end]
combo = pd.concat([df_train, oos_raw], ignore_index=True, sort=False)
combo = combo.sort_values('Date').reset_index(drop=True)
d = engineer_features(combo)
oos_mask_d = (d['Date'] > train_end).values

summary = []
detail_frames = []
for m in MARKETS:
    feats = pd.DataFrame(FEAT_FNS[m](d), index=d.index)
    keep = deployable_cols(feats, oos_mask_d)
    feats = feats[keep]
    print(f'[{m}] deployable features kept: {len(keep)}/{feats.shape[1] if False else len(FEAT_FNS[m](d))}')
    for h in HORIZONS:
        t0 = time.time()
        y = d[TARGET_LR[m]].shift(-1).rolling(h).sum().shift(-(h - 1))
        dat = pd.concat([feats, y.rename('y'), d['Date']], axis=1).dropna().reset_index(drop=True)
        Xall = dat[feats.columns]; yall = dat['y']; dates = dat['Date']
        oos_pos = np.where(dates.values > np.datetime64(train_end))[0]
        oos_pos = oos_pos[oos_pos >= 50]               # need history to train

        C = 0.01
        recs = []
        for k, i in enumerate(oos_pos):
            # train set: targets realised before origin i  -> positions <= i-h
            tr_end = i - h
            if tr_end < 40:
                continue
            if k % RETUNE_EVERY == 0:
                C = tune_C(Xall.iloc[:tr_end], yall.iloc[:tr_end])
            mtie = yall.iloc[:tr_end].values != 0
            Xtr = Xall.iloc[:tr_end][mtie]
            ytr = (np.sign(yall.iloc[:tr_end].values[mtie]) > 0).astype(int)
            if len(np.unique(ytr)) < 2:
                continue
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr)
            mdl = LogisticRegression(C=C, max_iter=500).fit(Xtr_s, ytr)
            prob = mdl.predict_proba(sc.transform(Xall.iloc[[i]]))[0, 1]
            sign_pred = 1.0 if prob > 0.5 else -1.0
            recs.append(dict(Date=dates.iloc[i], y=yall.iloc[i],
                             prob=prob, sign_pred=sign_pred))
        r = pd.DataFrame(recs)
        r['Market'] = m; r['Horizon'] = f'D+{h}'
        detail_frames.append(r)

        yv = r['y'].values; sp = r['sign_pred'].values
        nz = np.sign(yv) != 0
        da = (np.sign(yv[nz]) == sp[nz]).mean() * 100
        pt_s, pt_p = pt_test(yv[nz], sp[nz])
        pnl = sp * yv                      # hard long/short, gross
        ev = pnl.mean()
        sharpe = ev / (pnl.std(ddof=0) + 1e-12) * np.sqrt(252.0 / h)
        up_rate = (sp > 0).mean() * 100
        summary.append(dict(Market=m, Horizon=f'D+{h}', N=int(nz.sum()),
                            DA_pct=da, PT_p=pt_p, EV_bps=ev*1e4,
                            Sharpe=sharpe, pred_UP_pct=up_rate, C_last=C))
        print(f'  {m:5} D+{h}: DA={da:5.1f}%  PT_p={pt_p:.3f}  '
              f'EV={ev*1e4:+6.1f}bps  Sharpe*={sharpe:+5.2f}  '
              f'predUP={up_rate:3.0f}%  ({time.time()-t0:.0f}s)')

S = pd.DataFrame(summary)
S.to_csv('data/walkforward_directional_summary.csv', index=False)
pd.concat(detail_frames, ignore_index=True).to_csv(
    'data/walkforward_directional_detail.csv', index=False)

print('\n' + '=' * 92)
print('WALK-FORWARD DIRECTIONAL BACKTEST — expanding window, monthly C re-tune, deployable feats')
print('=' * 92)
with pd.option_context('display.float_format', lambda x: f'{x:.3f}'):
    print(S.to_string(index=False))
sig = S[S.PT_p < 0.10]
print('\nCells with PT p<0.10 (genuine directional ability):',
      ', '.join(f'{r.Market} {r.Horizon}' for _, r in sig.iterrows()) or 'NONE')
