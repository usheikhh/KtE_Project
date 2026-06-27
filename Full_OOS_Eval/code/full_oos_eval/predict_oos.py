"""Reconstructed OOS prediction pipeline for the K2E V4.4 deployed model.

Trains the V4.4 model on 2021-2024 (k2_clean_v4.csv), then generates
out-of-sample forecasts for every (market, horizon) cell over the OOS period,
applying the FIXED deployed policy (selected on the 2021-2024 CV):

    TTF   h1 -> L3 Markov (k=3 s=2 FV)
    TTF   h5 -> L3 Markov (k=3 s=3 FV)
    POWER h1 -> L3 Markov (k=5 s=2 SV)
    all other cells -> naive (random walk, lr=0)

Plus the EUA h=3 directional sign classifier (full-data logistic).

Validated against the original Jan-2025 oos_predictions.csv.
"""
import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from K2E_Model_V4_2 import (
    load_clean, engineer_features, make_X_y, FEAT_FNS,
    fit_layer1, fit_layer3, _order_states, _smoothed_probs_array,
    TARGET_P, TARGET_LR,
)
from K2E_Model_V4_3 import _prepare_sign_train, tune_logit_C

# Deployed L3 configs (market, h) -> (k_feats, n_states, switching_variance)
L3_CONFIGS = {
    ('TTF', 1):   dict(k_feats=3, n_states=2, switching_variance=False),
    ('TTF', 5):   dict(k_feats=3, n_states=3, switching_variance=False),
    ('POWER', 1): dict(k_feats=5, n_states=2, switching_variance=True),
}
MARKETS = ['TTF', 'POWER', 'EUA']
HORIZONS = [1, 3, 5]


def _filtered_last_probs(l3):
    """Causal regime seed: filtered P(state | data<=T), reordered to match
    the ordered state_betas/trans_mat."""
    res = l3['result']
    filt = np.asarray(res.filtered_marginal_probabilities, dtype=float)
    if filt.ndim == 1:
        filt = filt.reshape(-1, 1)
    order = l3['state_order']
    return filt[-1][order]


def fit_models(train_path):
    print('Loading + engineering training features (2021-2024)...')
    df_train = load_clean(train_path)
    d_train = engineer_features(df_train)

    l3_fits = {}
    for (mkt, h), cfg in L3_CONFIGS.items():
        print(f'  Fitting L3 {mkt} h={h}  (k={cfg["k_feats"]} s={cfg["n_states"]} '
              f'{"SV" if cfg["switching_variance"] else "FV"}) ...')
        l1 = fit_layer1(d_train, mkt, h)
        # build best_config with the cv fields fit_layer3 expects
        X, y = make_X_y(d_train, mkt, h)
        best_config = dict(cfg, cv_rmse=np.nan, cv_da=np.nan,
                           y_pred=np.zeros(len(y)), y_true=y.values)
        l3 = fit_layer3(d_train, mkt, h, l1, best_config)
        if l3 is None:
            raise RuntimeError(f'L3 final fit failed for {mkt} h={h}')
        l3['seed'] = _filtered_last_probs(l3)
        l3_fits[(mkt, h)] = l3

    # EUA h=3 deployable sign classifier (full-data logistic)
    print('  Fitting EUA h=3 directional sign classifier ...')
    Xe, ye = make_X_y(d_train, 'EUA', 3)
    Xtr, ytr_bin = _prepare_sign_train(Xe, ye)
    C_best = tune_logit_C(Xe, ye)
    sc_sign = StandardScaler()
    sign_mdl = LogisticRegression(penalty='l2', C=C_best, max_iter=500, solver='lbfgs')
    sign_mdl.fit(sc_sign.fit_transform(Xtr), ytr_bin)
    sign_pack = dict(model=sign_mdl, scaler=sc_sign, feats=Xe.columns.tolist(), C=C_best)

    return df_train, d_train, l3_fits, sign_pack


def _markov_predict_const(l3, X_scaled_rows, seed):
    """Predict cumulative log-return for each OOS row using a constant regime
    distribution = filtered last probs (mirrors CV: one prob vector applied to
    the whole forecast block)."""
    n_states = l3['n_states']
    betas = np.vstack([l3['state_betas'][k] for k in range(n_states)])  # (s, 1+k)
    Xc = np.column_stack([np.ones(len(X_scaled_rows)), X_scaled_rows])   # (n, 1+k)
    state_preds = Xc @ betas.T                                          # (n, s)
    return state_preds @ seed[:n_states]


def predict_oos(df_train, l3_fits, sign_pack, oos_raw, tail=80):
    """oos_raw: parsed OOS rows (Date > train end), k2_clean base schema."""
    # seed rolling windows with the training tail (raw level)
    combo = pd.concat([df_train.tail(tail), oos_raw], ignore_index=True, sort=False)
    combo = combo.sort_values('Date').reset_index(drop=True)
    d = engineer_features(combo)

    oos_dates = pd.to_datetime(oos_raw['Date']).values
    oos_mask = d['Date'].isin(oos_dates).values
    out = pd.DataFrame({'Date': d.loc[oos_mask, 'Date'].values})

    # actual closes at origin (for price reconstruction)
    closes = {m: d.loc[oos_mask, TARGET_P[m]].values for m in MARKETS}

    for m in MARKETS:
        for h in HORIZONS:
            if (m, h) in l3_fits:
                l3 = l3_fits[(m, h)]
                feats_all = pd.DataFrame(FEAT_FNS[m](d), index=d.index)
                Xrows = feats_all.loc[oos_mask, l3['top_feats']]
                Xs = np.nan_to_num(l3['scaler'].transform(Xrows.values), nan=0.0)
                lr = _markov_predict_const(l3, Xs, l3['seed'])
                # rows with NaN features -> fall back to naive (0)
                lr = np.where(np.isfinite(lr), lr, 0.0)
            else:
                lr = np.zeros(oos_mask.sum())  # naive
            price = closes[m] * np.exp(lr)
            out[f'{m}_h{h}_lr'] = lr
            out[f'{m}_h{h}_price'] = price

    # EUA h=3 directional signal
    feats_e = pd.DataFrame(FEAT_FNS['EUA'](d), index=d.index)
    Xe = feats_e.loc[oos_mask, sign_pack['feats']]
    Xe_s = np.nan_to_num(sign_pack['scaler'].transform(Xe.values), nan=0.0)
    prob = sign_pack['model'].predict_proba(Xe_s)[:, 1]
    out['EUA_h3_sign_signal'] = 2.0 * prob - 1.0
    out['EUA_h3_direction'] = np.where(prob > 0.5, 'UP', 'DOWN')
    return out


if __name__ == '__main__':
    TRAIN = 'data/k2_clean_v4.csv'
    df_train, d_train, l3_fits, sign_pack = fit_models(TRAIN)

    full = pd.read_csv('data/full_parsed.csv', parse_dates=['Date'])
    train_end = pd.to_datetime(df_train['Date']).max()
    oos_raw = full[full['Date'] > train_end].reset_index(drop=True)
    print(f'\nOOS rows: {len(oos_raw)}  ({oos_raw.Date.min().date()} -> {oos_raw.Date.max().date()})')

    pred = predict_oos(df_train, l3_fits, sign_pack, oos_raw)
    pred.to_csv('data/full_oos_predictions.csv', index=False)
    print(f'Saved full_oos_predictions.csv: {len(pred)} rows')

    # ---- VALIDATION against original Jan-2025 predictions ----
    ref = pd.read_csv('data/oos_predictions.csv', parse_dates=['Date'])
    m = ref.merge(pred, on='Date', suffixes=('_ref', '_new'))
    print(f'\nVALIDATION vs original oos_predictions.csv ({len(m)} matched dates):')
    for col in ['TTF_h1_lr', 'TTF_h5_lr', 'POWER_h1_lr', 'EUA_h3_sign_signal']:
        a, b = m[f'{col}_ref'], m[f'{col}_new']
        print(f'  {col:22} maxabs diff = {(a-b).abs().max():.6g}   '
              f'corr = {a.corr(b):.4f}')
    dir_match = (m['EUA_h3_direction_ref'] == m['EUA_h3_direction_new']).mean()
    print(f'  EUA_h3_direction       agreement = {100*dir_match:.1f}%')
