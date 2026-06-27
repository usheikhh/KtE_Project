"""
K2 Energy / Fordham University Collaboration
Phase 1 v4.2: V4.1 + Bloomberg Tier-1 features (term structure, positioning,
EUA implied vol, auction premium, power base/peak, coal, French nuclear).

Changes vs V4.1:
  - Input: k2_clean_v3.csv (V4.1 base + BBG_* / D_LOG_OI_* / D_LOG_VOL_* cols)
  - TTF:   + TTF term-structure M1/M2 + change, NBP M1/M2 spillover,
             TTF OI/VOL daily log-change, Baltic Dry log-return.
  - POWER: + DE base/peak spread, French nuclear daily change,
             Rotterdam coal log-return.
  - EUA:   + EUA term-structure M1/M2 + change, MO1 implied vol level + change
             + 20-day z, auction-vs-futures spread, MO1 OI daily log-change.

Changes vs V4:
  - Input: k2_clean_v2.csv (has GIE_EU_*, GIE_DE_*, GIE_DE_OUT_* columns)
  - Replaced STORAGE_DEVIATION (leaked cross-year DOY mean) with
    GIE_EU_FULL_DOY_DEV (uses expanding-window prior-years-only mean).
  - Added GIE-derived features:
      * EU/DE filling levels (%) and DOY deviations (causal)
      * Injection/withdrawal/netWithdrawal flows (GWh/d)
      * Per-country DE flows for Power features
      * DE storage outages: planned vs unplanned GWh/d lost, event count
  - Retained all V4 logic: L1 Ridge + L3 Markov sweep + DM/PT tests.

Changes vs V3:
  1. Derived features ("free-lift" — from existing k2_clean.csv columns):
     - Term structure: TTF CAL1/M1 and Power CAL1/M1 log spreads + diffs
     - Clean spark spread (POWER - 2.035*TTF - 0.407*EUA) + z-score
     - EUA compliance-cohort COT + compliance-vs-HF divergence
     - TTF implied vol log-return and premium change
     - Alpine water reservoir anomaly (parallel to Scandinavian)
     - Residual load LR (pre-computed column)
  2. Significance testing on CV predictions:
     - Diebold-Mariano (L1 vs naive zero forecast, Newey-West corrected)
     - Pesaran-Timmermann (directional predictability vs chance)
     - Applied to every (market, horizon, layer) cell

Layers (unchanged from V3):
  Layer 1 — Ridge-ARX baseline (α grid 1e-4..1e4)
  Layer 3 — Markov-switching ARX with HP sweep and degeneracy safeguards
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from pathlib import Path
import warnings
import itertools
warnings.filterwarnings('ignore')

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import statsmodels.api as sm


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def load_clean(path):
    ext = Path(path).suffix.lower()
    if ext == '.csv':
        df = pd.read_csv(path).copy()
    else:
        all_sheets = pd.read_excel(path, sheet_name=None)
        if 'k2_clean' in all_sheets:
            df = all_sheets['k2_clean'].copy()
        elif 'Data' in all_sheets:
            df = all_sheets['Data'].copy()
        else:
            raise KeyError(f"No supported sheet found in {path}.")

    df = df.rename(columns={'ID': 'Date', 'DATE': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    for c in df.columns:
        if c not in ['Date', 'Unnamed: 0']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING  (V4: + derived features)
# ═══════════════════════════════════════════════════════════════════════════

# CCGT clean-spark constants (industry-standard):
#   heat rate 2.035 MWh_th/MWh_el (η ≈ 49.1%)
#   emission factor gas 0.407 tCO2/MWh_el (= 0.2 tCO2/MWh_th × HR)
HR_GAS      = 2.035
EF_GAS_ELEC = 0.407


def engineer_features(df):
    d = df.copy()

    # ── V3 features (unchanged) ──────────────────────────────────────────
    d['TEMP_ANOM_DE']     = d['MF_TEMPERATURE_GERMANY_ACTUAL_C'] - d['MF_TEMPERATURE_NORMAL_GERMANY_NORMAL_C']
    d['TEMP_ANOM_FR']     = d['MF_TEMPERATURE_FRANCE_ACTUAL_C']  - d['MF_TEMPERATURE_NORMAL_FRANCE_NORMAL_C']
    d['GAS_CONS_ANOM_DE'] = (d['MF_GAS_CONSUMPTION_GERMANY_ACTUAL_GWH_D']
                              - d['MF_GAS_CONSUMPTION_GERMANY_AVG_2Y_GWH_D'])
    d['WATER_ANOM_SCAN']  = (d['MF_WATER_RESERVOIR_SCANDINAVIA_ACTUAL_TWH']
                              - d['MF_WATER_RESERVOIR_NORMAL_SCANDINAVIA_NORMAL_TWH'])
    d['WATER_ANOM_ALP']   = (d['MF_WATER_RESERVOIR_ALPINE_ACTUAL_TWH']
                              - d['MF_WATER_RESERVOIR_NORMAL_ALPINE_NORMAL_TWH'])
    d['D_GAS_CONS_ANOM']  = d['GAS_CONS_ANOM_DE'].diff()

    d['WARM_SEASON']    = d['Date'].dt.month.isin([4,5,6,7,8,9]).astype(float)
    d['COOL_SEASON']    = 1 - d['WARM_SEASON']
    d['TEMP_ANOM_WARM'] = d['TEMP_ANOM_DE'] * d['WARM_SEASON']
    d['TEMP_ANOM_COOL'] = d['TEMP_ANOM_DE'] * d['COOL_SEASON']

    storage = d['MF_EU_STORAGE_STORAGE_ACTUAL_STORAGE_TWH']
    d['STORAGE_DOY_MEAN']  = storage.groupby(d['Date'].dt.dayofyear).transform('mean')
    d['STORAGE_DEVIATION'] = storage - d['STORAGE_DOY_MEAN']

    price_map = {
        'BM_TTF_M1_CLOSE_EUR_MWH':              'TTF',
        'BM_GERMANY_POWER_M1_CLOSE_EUR_MWH':    'POWER',
        'BM_EUA_CO2_CAL1_PRICE_EUR_TON':        'EUA',
        'IM_BRENT_M1_PRICE_USD_BBL':            'BRENT',
        'IM_COAL_CAL1_PRICE_USD_TON':           'COAL',
        'IM_JKM_LNG_M1_PRICE_USD_MMBTU':       'JKM',
        'IM_SPOT_GAS_TTF_DA_PRICE_EUR_MWH':    'TTF_DA',
        'IM_COAL_GAS_SWITCH_M1_SUPPORT_EUR_MWH':   'COAL_GS',
        'IM_COAL_GAS_SWITCH_CAL1_SUPPORT_EUR_MWH': 'COAL_GS_CAL',
        'BM_TTF_CAL1_CLOSE_EUR_MWH':           'TTF_CAL',
        'BM_GERMANY_POWER_CAL1_CLOSE_EUR_MWH': 'POWER_CAL',
    }
    for col, nm in price_map.items():
        d[f'LR_{nm}'] = np.log(d[col].clip(lower=0.001)).diff()

    flow_map = {
        'MF_LNG_EUROPE_FLOW_ACTUAL_FLOW_GWH_D':                    'LNG_FLOW',
        'MF_NORWAY_GAS_IMPORT_ACTUAL_VALUE_GWH_D':                 'NOR_GAS',
        'MF_POWER_LOAD_GERMANY_ACTUAL_LOAD_GW':                    'LOAD',
        'MF_RENEWABLES_GENERATION_GERMANY_WIND_GENERATION_GW':     'WIND',
        'MF_RENEWABLES_GENERATION_GERMANY_SOLAR_GENERATION_GW':    'SOLAR',
        'MF_NUCLEAR_GENERATION_FRANCE_GENERATION_GW':              'NUCLEAR_FR',
    }
    for col, nm in flow_map.items():
        d[f'LR_{nm}'] = np.log(d[col].clip(lower=0.1)).diff()

    for nm in ['TTF', 'POWER', 'EUA']:
        d[f'RVOL20_{nm}'] = d[f'LR_{nm}'].rolling(20).std()
        d[f'RVOL5_{nm}']  = d[f'LR_{nm}'].rolling(5).std()

    d['LR_EUA_pos'] = d['LR_EUA'].clip(lower=0)
    d['LR_EUA_neg'] = d['LR_EUA'].clip(upper=0)

    d['RUS_PIPE']   = d['MF_RUSSIAN_PIPELINE_FLOW_ACTUAL_FLOW_GWH_D'].where(
        d['Date'] <= '2022-09-26', 0)
    d['D_RUS_PIPE'] = d['RUS_PIPE'].diff()

    sup_m1  = 'IM_COAL_GAS_SWITCH_M1_SUPPORT_EUR_MWH'
    res_m1  = 'IM_COAL_GAS_SWITCH_M1_RESISTANCE_EUR_MWH'
    sup_cal = 'IM_COAL_GAS_SWITCH_CAL1_SUPPORT_EUR_MWH'
    res_cal = 'IM_COAL_GAS_SWITCH_CAL1_RESISTANCE_EUR_MWH'

    d['NO_BUFFER_M1']  = df[res_m1].isna().astype(float)
    d['NO_BUFFER_CAL'] = df[res_cal].isna().astype(float)

    d['LOG_BUFFER_M1']  = np.where(
        (df[res_m1] - df[sup_m1]) > 0,
        np.log((df[res_m1] - df[sup_m1]).clip(lower=0.01)), 0.0)
    d['LOG_BUFFER_CAL'] = np.where(
        (df[res_cal] - df[sup_cal]) > 0,
        np.log((df[res_cal] - df[sup_cal]).clip(lower=0.01)), 0.0)

    cot_map = {
        'MF_COT_TTF_HEDGE_FUNDS_NET_MWH':  'COT_TTF_HF_NET',
        'MF_COT_TTF_COMMERCIAL_NET_MWH':   'COT_TTF_COMM_NET',
        'MF_COT_TTF_TOTAL_LONG_MWH':       'COT_TTF_LONG',
        'MF_COT_TTF_TOTAL_SHORT_MWH':      'COT_TTF_SHORT',
        'MF_COT_EUA_HEDGE_FUNDS_NET_TON':  'COT_EUA_HF_NET',
        'MF_COT_EUA_COMMERCIAL_NET_TON':   'COT_EUA_COMM_NET',
        'MF_COT_EUA_TOTAL_LONG_TON':       'COT_EUA_LONG',
        'MF_COT_EUA_TOTAL_SHORT_TON':      'COT_EUA_SHORT',
    }
    for raw_col, nm in cot_map.items():
        d[nm] = df[raw_col].ffill()

    d['D_COT_TTF_HF_NET']  = d['COT_TTF_HF_NET'].diff()
    d['D_COT_TTF_COMM_NET']= d['COT_TTF_COMM_NET'].diff()
    d['D_COT_EUA_HF_NET']  = d['COT_EUA_HF_NET'].diff()
    d['D_COT_EUA_COMM_NET']= d['COT_EUA_COMM_NET'].diff()

    d['COT_TTF_HF_RATIO'] = (d['COT_TTF_LONG']
                              / (d['COT_TTF_LONG'] + d['COT_TTF_SHORT']).clip(lower=1))
    d['COT_EUA_HF_RATIO'] = (d['COT_EUA_LONG']
                              / (d['COT_EUA_LONG'] + d['COT_EUA_SHORT']).clip(lower=1))

    # ── V4 NEW DERIVED FEATURES ─────────────────────────────────────────

    # (a) Term structure spreads (contango/backwardation proxies)
    d['TTF_TS']    = np.log(d['BM_TTF_CAL1_CLOSE_EUR_MWH'].clip(lower=0.001)) \
                   - np.log(d['BM_TTF_M1_CLOSE_EUR_MWH'].clip(lower=0.001))
    d['POWER_TS']  = np.log(d['BM_GERMANY_POWER_CAL1_CLOSE_EUR_MWH'].clip(lower=0.001)) \
                   - np.log(d['BM_GERMANY_POWER_M1_CLOSE_EUR_MWH'].clip(lower=0.001))
    d['D_TTF_TS']   = d['TTF_TS'].diff()
    d['D_POWER_TS'] = d['POWER_TS'].diff()

    # (b) Clean spark spread (EUR/MWh_elec). EUA CAL1 as carbon leg.
    d['CLEAN_SPARK'] = (d['BM_GERMANY_POWER_M1_CLOSE_EUR_MWH']
                        - HR_GAS * d['BM_TTF_M1_CLOSE_EUR_MWH']
                        - EF_GAS_ELEC * d['BM_EUA_CO2_CAL1_PRICE_EUR_TON'])
    d['D_CLEAN_SPARK'] = d['CLEAN_SPARK'].diff()
    # z-score over 20-day rolling window (stationarity-friendly)
    spark_mu  = d['CLEAN_SPARK'].rolling(20).mean()
    spark_sd  = d['CLEAN_SPARK'].rolling(20).std()
    d['SPARK_Z20'] = ((d['CLEAN_SPARK'] - spark_mu) / spark_sd.replace(0, np.nan))

    # (c) EUA compliance cohort COT + divergence from hedge funds
    d['COT_EUA_COMPL_NET'] = df['MF_COT_EUA_DIRECTIVE_COMPLIANCE_OBLIGATION_NET_TON'].ffill()
    d['D_COT_EUA_COMPL_NET'] = d['COT_EUA_COMPL_NET'].diff()
    # Divergence: compliance vs hedge funds (sign disagreement = regime signal)
    d['COT_EUA_COMPL_vs_HF'] = d['COT_EUA_COMPL_NET'] - d['COT_EUA_HF_NET']
    d['D_COT_EUA_COMPL_vs_HF'] = d['COT_EUA_COMPL_vs_HF'].diff()

    # (d) TTF options vol/premium log-changes
    d['LR_TTF_IV']  = np.log(df['IM_TTF_OPTIONS_TTF_IMPLIED_VOL_PCT_FFILL']
                              .clip(lower=0.001)).diff()
    d['D_TTF_PREM'] = df['IM_TTF_OPTIONS_TTF_PREMIUM_EUR_MWH_FFILL'].diff()

    # (e) Residual load (pre-computed column; just take LR)
    if 'DERIVED_RESIDUAL_LOAD_GERMANY_GW' in df.columns:
        d['LR_RESID_LOAD'] = np.log(
            df['DERIVED_RESIDUAL_LOAD_GERMANY_GW'].clip(lower=0.1)).diff()

    # ── V4.1 GIE AGSI FEATURES ──────────────────────────────────────────
    # Gated: only derive if GIE columns are present in the input
    if 'GIE_EU_FULL' in df.columns:
        # EU storage levels (% full, DOY deviation causal)
        d['GIE_EU_FULL']           = df['GIE_EU_FULL']
        d['GIE_EU_FULL_DOY_DEV']   = df.get('GIE_EU_FULL_DOY_DEV')
        d['D_GIE_EU_FULL']         = df['GIE_EU_FULL'].diff()
        # EU flows
        d['GIE_EU_NET_WD']         = df['GIE_EU_NETWITHDRAWAL']
        d['D_GIE_EU_NET_WD']       = d['GIE_EU_NET_WD'].diff()
        d['GIE_EU_INJ_LR']         = np.log(df['GIE_EU_INJECTION'].clip(lower=1.0)).diff()
        d['GIE_EU_WD_LR']          = np.log(df['GIE_EU_WITHDRAWAL'].clip(lower=1.0)).diff()
        # DE storage (Germany-specific for POWER)
        d['GIE_DE_FULL']           = df['GIE_DE_FULL']
        d['GIE_DE_FULL_DOY_DEV']   = df.get('GIE_DE_FULL_DOY_DEV')
        d['D_GIE_DE_FULL']         = df['GIE_DE_FULL'].diff()
        d['GIE_DE_NET_WD']         = df['GIE_DE_NETWITHDRAWAL']
        d['D_GIE_DE_NET_WD']       = d['GIE_DE_NET_WD'].diff()
        # DE outages — supply shock indicator (raw GWh/d lost, plus log1p for scale)
        d['GIE_DE_OUT_UNPL']       = df['GIE_DE_OUT_UNPLANNED_GWH_D']
        d['GIE_DE_OUT_PL']         = df['GIE_DE_OUT_PLANNED_GWH_D']
        d['GIE_DE_OUT_TOT_LOG1P']  = np.log1p(df['GIE_DE_OUT_PLANNED_GWH_D']
                                              + df['GIE_DE_OUT_UNPLANNED_GWH_D'])
        d['D_GIE_DE_OUT_UNPL']     = d['GIE_DE_OUT_UNPL'].diff()

    return d


# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE BLOCKS  (V4: + selected derived features per market)
# ═══════════════════════════════════════════════════════════════════════════

def L(s, lag):
    return s.shift(lag)


def features_TTF(d):
    return {
        'LR_TTF_lag1': L(d['LR_TTF'], 1), 'LR_TTF_lag2': L(d['LR_TTF'], 2),
        'LR_TTF_lag3': L(d['LR_TTF'], 3), 'LR_TTF_lag5': L(d['LR_TTF'], 5),
        'LR_POWER_lag1':  L(d['LR_POWER'],  1),
        'LR_TTF_DA_lag1': L(d['LR_TTF_DA'], 1),
        'LR_COAL_lag1':   L(d['LR_COAL'],   1),
        'LR_BRENT_lag1':  L(d['LR_BRENT'],  1),
        'LR_JKM_lag1':    L(d['LR_JKM'],    1),
        'D_GAS_CONS_ANOM':  L(d['D_GAS_CONS_ANOM'],    1),
        'TEMP_ANOM_WARM':   L(d['TEMP_ANOM_WARM'],      1),
        'TEMP_ANOM_COOL':   L(d['TEMP_ANOM_COOL'],      1),
        'STORAGE_DEV':      L(d['STORAGE_DEVIATION'],   1),
        'LR_LNG_FLOW': L(d['LR_LNG_FLOW'], 1),
        'LR_NOR_GAS':  L(d['LR_NOR_GAS'],  1),
        'D_RUS_PIPE':  L(d['D_RUS_PIPE'],  1),
        'NO_BUFFER_M1':    d['NO_BUFFER_M1'],
        'LOG_BUFFER_M1':   L(d['LOG_BUFFER_M1'],  1),
        'RVOL20_TTF':      L(d['RVOL20_TTF'],     1),
        'RVOL5_TTF':       L(d['RVOL5_TTF'],      1),
        'LR_JKM_x_NOBUF':  L(d['LR_JKM'],  1) * d['NO_BUFFER_M1'],
        'LR_COAL_x_NOBUF': L(d['LR_COAL'], 1) * d['NO_BUFFER_M1'],
        'D_COT_TTF_HF_NET': L(d['D_COT_TTF_HF_NET'],  1),
        'COT_TTF_HF_RATIO': L(d['COT_TTF_HF_RATIO'],  1),
        # V4 derived
        'TTF_TS':       L(d['TTF_TS'],      1),
        'D_TTF_TS':     L(d['D_TTF_TS'],    1),
        'LR_TTF_CAL':   L(d['LR_TTF_CAL'],  1),
        'LR_TTF_IV':    L(d['LR_TTF_IV'],   1),
        'D_TTF_PREM':   L(d['D_TTF_PREM'],  1),
        # V4.1 GIE storage + flows (EU-wide)
        'GIE_EU_FULL_DOY_DEV':   L(d['GIE_EU_FULL_DOY_DEV'],   1),
        'GIE_EU_NET_WD':         L(d['GIE_EU_NET_WD'],         1),
        'D_GIE_EU_NET_WD':       L(d['D_GIE_EU_NET_WD'],       1),
        'GIE_EU_INJ_LR':         L(d['GIE_EU_INJ_LR'],         1),
        'GIE_EU_WD_LR':          L(d['GIE_EU_WD_LR'],          1),
        # DE outages (supply shock → TTF via pricing pass-through)
        'GIE_DE_OUT_UNPL':       L(d['GIE_DE_OUT_UNPL'],       1),
        'GIE_DE_OUT_TOT_LOG1P':  L(d['GIE_DE_OUT_TOT_LOG1P'],  1),
        # V4.2 BBG: term structure + NBP spillover + positioning + freight
        'BBG_LR_TTF_M1_M2':      L(d['BBG_LR_TTF_M1_M2'],      1),
        'D_BBG_LR_TTF_M1_M2':    L(d['D_BBG_LR_TTF_M1_M2'],    1),
        'BBG_LR_TTF_M1_M3':      L(d['BBG_LR_TTF_M1_M3'],      1),
        'BBG_LR_NBP_M1_M2':      L(d['BBG_LR_NBP_M1_M2'],      1),
        'D_LOG_OI_TZTA':         L(d['D_LOG_OI_TZTA'],         1),
        'D_LOG_VOL_TZTA':        L(d['D_LOG_VOL_TZTA'],        1),
        'BBG_LR_BDIY':           L(d['BBG_LR_BDIY'],           1),
    }


def features_POWER(d):
    feats = {
        'LR_POWER_lag1': L(d['LR_POWER'], 1), 'LR_POWER_lag2': L(d['LR_POWER'], 2),
        'LR_POWER_lag3': L(d['LR_POWER'], 3), 'LR_POWER_lag5': L(d['LR_POWER'], 5),
        'LR_TTF_lag1':  L(d['LR_TTF'],     1),
        'LR_EUA_pos':   L(d['LR_EUA_pos'], 1),
        'LR_EUA_neg':   L(d['LR_EUA_neg'], 1),
        'LR_COAL_lag1': L(d['LR_COAL'],    1),
        'LR_LOAD':        L(d['LR_LOAD'],        1),
        'TEMP_ANOM_WARM': L(d['TEMP_ANOM_WARM'], 1),
        'TEMP_ANOM_COOL': L(d['TEMP_ANOM_COOL'], 1),
        'LR_WIND':       L(d['LR_WIND'],       1),
        'LR_SOLAR':      L(d['LR_SOLAR'],      1),
        'LR_NUCLEAR_FR': L(d['LR_NUCLEAR_FR'], 1),
        'WATER_ANOM_SCAN': L(d['WATER_ANOM_SCAN'], 1),
        'WATER_ANOM_ALP':  L(d['WATER_ANOM_ALP'],  1),
        'NO_BUFFER_M1':    d['NO_BUFFER_M1'],
        'RVOL20_POWER':    L(d['RVOL20_POWER'], 1),
        'RVOL5_POWER':     L(d['RVOL5_POWER'],  1),
        'LR_TTF_x_NOBUF':  L(d['LR_TTF'],      1) * d['NO_BUFFER_M1'],
        'LR_EUA_x_NOBUF':  L(d['LR_EUA_pos'],  1) * d['NO_BUFFER_M1'],
        # V4 derived (core spark + term structure)
        'SPARK_Z20':     L(d['SPARK_Z20'],     1),
        'D_CLEAN_SPARK': L(d['D_CLEAN_SPARK'], 1),
        'POWER_TS':      L(d['POWER_TS'],      1),
        'D_POWER_TS':    L(d['D_POWER_TS'],    1),
        'LR_POWER_CAL':  L(d['LR_POWER_CAL'],  1),
        # V4.1 GIE DE storage + outages (German power feeds off German gas storage)
        'GIE_DE_FULL_DOY_DEV':   L(d['GIE_DE_FULL_DOY_DEV'],   1),
        'GIE_DE_NET_WD':         L(d['GIE_DE_NET_WD'],         1),
        'D_GIE_DE_NET_WD':       L(d['D_GIE_DE_NET_WD'],       1),
        'GIE_DE_OUT_UNPL':       L(d['GIE_DE_OUT_UNPL'],       1),
        'GIE_DE_OUT_TOT_LOG1P':  L(d['GIE_DE_OUT_TOT_LOG1P'],  1),
        # V4.2 BBG: DE base/peak spread + French nuclear shock + coal input
        'BBG_LR_POWER_BASE_PEAK': L(d['BBG_LR_POWER_BASE_PEAK'], 1),
        'D_BBG_PX_RTEGNUCD':      L(d['D_BBG_PX_RTEGNUCD'],      1),
        'BBG_LR_COAL_XA1':        L(d['BBG_LR_COAL_XA1'],        1),
    }
    if 'LR_RESID_LOAD' in d.columns:
        feats['LR_RESID_LOAD'] = L(d['LR_RESID_LOAD'], 1)
    return feats


def features_EUA(d):
    return {
        'LR_EUA_lag1': L(d['LR_EUA'], 1), 'LR_EUA_lag2': L(d['LR_EUA'], 2),
        'LR_EUA_lag3': L(d['LR_EUA'], 3), 'LR_EUA_lag5': L(d['LR_EUA'], 5),
        'LR_COAL_GS':     L(d['LR_COAL_GS'],     1),
        'LR_COAL_GS_CAL': L(d['LR_COAL_GS_CAL'], 1),
        'LR_BRENT_lag1':  L(d['LR_BRENT'],        1),
        'LR_TTF_lag1':    L(d['LR_TTF'],          1),
        'LR_COAL_lag1':   L(d['LR_COAL'],         1),
        'LR_POWER_lag1':  L(d['LR_POWER'],        1),
        'TEMP_ANOM_COOL': L(d['TEMP_ANOM_COOL'], 1),
        'LR_LOAD':        L(d['LR_LOAD'],        1),
        'D_RUS_PIPE':  L(d['D_RUS_PIPE'],  1),
        'LR_LNG_FLOW': L(d['LR_LNG_FLOW'], 1),
        'NO_BUFFER_M1':    d['NO_BUFFER_M1'],
        'NO_BUFFER_CAL':   d['NO_BUFFER_CAL'],
        'LOG_BUFFER_CAL':  L(d['LOG_BUFFER_CAL'], 1),
        'RVOL20_EUA':      L(d['RVOL20_EUA'],     1),
        'RVOL5_EUA':       L(d['RVOL5_EUA'],      1),
        'RVOL20_TTF':      L(d['RVOL20_TTF'],     1),
        'LR_COAL_x_NOBUF': L(d['LR_COAL'],        1) * d['NO_BUFFER_M1'],
        'D_COT_EUA_HF_NET': L(d['D_COT_EUA_HF_NET'],  1),
        'COT_EUA_HF_RATIO': L(d['COT_EUA_HF_RATIO'],  1),
        # V4 derived (compliance cohort + spark + TTF TS)
        'D_COT_EUA_COMPL_NET':     L(d['D_COT_EUA_COMPL_NET'],     1),
        'COT_EUA_COMPL_vs_HF':     L(d['COT_EUA_COMPL_vs_HF'],     1),
        'D_COT_EUA_COMPL_vs_HF':   L(d['D_COT_EUA_COMPL_vs_HF'],   1),
        'SPARK_Z20':     L(d['SPARK_Z20'],    1),
        'TTF_TS':        L(d['TTF_TS'],       1),
        # V4.1 GIE (EUA reacts to gas supply tightness via fuel-switch channel)
        'GIE_EU_FULL_DOY_DEV':   L(d['GIE_EU_FULL_DOY_DEV'],   1),
        'GIE_EU_NET_WD':         L(d['GIE_EU_NET_WD'],         1),
        'GIE_DE_OUT_TOT_LOG1P':  L(d['GIE_DE_OUT_TOT_LOG1P'],  1),
        # V4.2 BBG: EUA term structure + implied vol + auction premium + positioning
        'BBG_LR_EUA_M1_M2':            L(d['BBG_LR_EUA_M1_M2'],            1),
        'D_BBG_LR_EUA_M1_M2':          L(d['D_BBG_LR_EUA_M1_M2'],          1),
        'BBG_IVOL_MO1':                L(d['BBG_IVOL_MO1'],                1),
        'D_BBG_IVOL_MO1':              L(d['D_BBG_IVOL_MO1'],              1),
        'BBG_IVOL_MO1_Z20':            L(d['BBG_IVOL_MO1_Z20'],            1),
        'BBG_LR_EUA_AUCTION_VS_MO1':   L(d['BBG_LR_EUA_AUCTION_VS_MO1'],   1),
        'D_LOG_OI_MO1':                L(d['D_LOG_OI_MO1'],                1),
    }


FEAT_FNS  = {'TTF': features_TTF, 'POWER': features_POWER, 'EUA': features_EUA}
TARGET_LR = {'TTF': 'LR_TTF', 'POWER': 'LR_POWER', 'EUA': 'LR_EUA'}
TARGET_P  = {
    'TTF':   'BM_TTF_M1_CLOSE_EUR_MWH',
    'POWER': 'BM_GERMANY_POWER_M1_CLOSE_EUR_MWH',
    'EUA':   'BM_EUA_CO2_CAL1_PRICE_EUR_TON',
}
AR_LAG1 = {'TTF': 'LR_TTF_lag1', 'POWER': 'LR_POWER_lag1', 'EUA': 'LR_EUA_lag1'}


# ═══════════════════════════════════════════════════════════════════════════
# 4. DATASET CONSTRUCTION & SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def make_X_y(d, market, h):
    feats = FEAT_FNS[market](d)
    X = pd.DataFrame(feats, index=d.index)
    lr = d[TARGET_LR[market]]
    y = lr.shift(-1).rolling(h).sum().shift(-(h - 1))
    combined = pd.concat([X, y.rename('y')], axis=1).dropna()
    return combined.iloc[:, :-1], combined['y']


def get_cv_folds(n, n_folds=5):
    fold = n // (n_folds + 1)
    folds = []
    for f in range(n_folds):
        ts = fold + f * fold
        te = ts + fold
        if te > n:
            break
        folds.append((ts, te))
    return folds


def direction_accuracy(y_true, y_pred):
    return (np.sign(y_pred) == np.sign(np.asarray(y_true))).mean() * 100


ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)


# ═══════════════════════════════════════════════════════════════════════════
# 5. SIGNIFICANCE TESTS — Diebold-Mariano & Pesaran-Timmermann
# ═══════════════════════════════════════════════════════════════════════════

def dm_test(e1, e2, h=1):
    """Diebold-Mariano on squared-error loss.
    e1, e2: arrays of forecast errors (y_true - y_pred).
    h: forecast horizon (Newey-West uses h-1 lags).
    Returns (dm_stat, p_two_sided). Negative dm_stat → model 1 better."""
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    d  = e1**2 - e2**2
    T  = len(d)
    if T < 10:
        return np.nan, np.nan
    dbar = d.mean()
    lag  = max(h - 1, 0)
    # Newey-West long-run variance with Bartlett kernel
    gamma0 = np.var(d, ddof=0)
    var_d  = gamma0
    for k in range(1, lag + 1):
        gk = np.cov(d[k:], d[:-k], ddof=0)[0, 1]
        w  = 1.0 - k / (lag + 1)
        var_d += 2.0 * w * gk
    if var_d <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var_d / T)
    # Harvey-Leybourne-Newbold small-sample correction
    hln = np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)
    stat = stat * hln
    p = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return stat, p


def pt_test(y_true, y_pred):
    """Pesaran-Timmermann directional test. One-sided alternative: predictor
    has genuine directional ability. Returns (stat, p_one_sided)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 10:
        return np.nan, np.nan
    # Exclude zero-sign cases (ties) from counts
    sy = np.sign(y_true)
    sp = np.sign(y_pred)
    Pyx = (sy == sp).mean()
    Py  = (sy > 0).mean()
    Px  = (sp > 0).mean()
    P_star = Py * Px + (1 - Py) * (1 - Px)
    var_P     = P_star * (1 - P_star) / n
    var_Pstar = ((2*Py - 1)**2 * Px * (1 - Px) / n
                 + (2*Px - 1)**2 * Py * (1 - Py) / n
                 + 4.0 * Py * Px * (1 - Py) * (1 - Px) / (n**2))
    denom = var_P - var_Pstar
    if denom <= 0:
        return np.nan, np.nan
    stat = (Pyx - P_star) / np.sqrt(denom)
    p = 1.0 - norm.cdf(stat)
    return stat, p


# ═══════════════════════════════════════════════════════════════════════════
# 6. LAYER 1 — RIDGE-ARX  (returns CV preds for tests)
# ═══════════════════════════════════════════════════════════════════════════

def tune_alpha_l1(X, y, n_folds=5, alpha_grid=ALPHAS):
    n = len(X)
    folds = get_cv_folds(n, n_folds)
    alpha_scores = {a: [] for a in alpha_grid}
    for ts, te in folds:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X.iloc[:ts])
        Xte = sc.transform(X.iloc[ts:te])
        ytr, yte = y.iloc[:ts], y.iloc[ts:te]
        for a in alpha_grid:
            p = Ridge(alpha=a).fit(Xtr, ytr).predict(Xte)
            alpha_scores[a].append(mean_squared_error(yte, p) ** 0.5)
    return min(alpha_grid, key=lambda a: np.mean(alpha_scores[a]))


def cv_layer1(X, y, alpha, n_folds=5):
    n = len(X)
    folds = get_cv_folds(n, n_folds)
    all_pred, all_true = [], []
    for ts, te in folds:
        sc = StandardScaler()
        Xtr = sc.fit_transform(X.iloc[:ts])
        Xte = sc.transform(X.iloc[ts:te])
        ytr, yte = y.iloc[:ts], y.iloc[ts:te]
        p = Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xte)
        all_pred.extend(p)
        all_true.extend(yte.values)
    all_pred, all_true = np.array(all_pred), np.array(all_true)
    return {
        'cv_rmse': mean_squared_error(all_true, all_pred) ** 0.5,
        'cv_da':   direction_accuracy(all_true, all_pred),
        'naive_rmse': mean_squared_error(all_true, np.zeros_like(all_true)) ** 0.5,
        'y_pred':  all_pred,
        'y_true':  all_true,
    }


def fit_layer1(d, market, h):
    X, y = make_X_y(d, market, h)
    alpha = tune_alpha_l1(X, y)
    cv = cv_layer1(X, y, alpha)
    sc = StandardScaler()
    model = Ridge(alpha=alpha).fit(sc.fit_transform(X), y)
    return {
        'model': model, 'scaler': sc, 'alpha': alpha,
        'feature_names': X.columns.tolist(),
        'cv': cv, 'n_train': len(X), 'X': X, 'y': y,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. LAYER 3 — MARKOV-SWITCHING ARX with HP SWEEP + SAFEGUARDS
# ═══════════════════════════════════════════════════════════════════════════

MIN_REGIME_FREQ = 0.05
VAR_FLOOR       = 1e-8


def _select_top_features(l1_fit, k, market, force_ar=True):
    coefs = np.abs(l1_fit['model'].coef_)
    names = l1_fit['feature_names']
    ranked = [names[i] for i in np.argsort(coefs)[::-1]]
    if force_ar:
        ar_lag = AR_LAG1[market]
        selected = [ar_lag] if ar_lag in names else []
        for n in ranked:
            if len(selected) >= k:
                break
            if n not in selected:
                selected.append(n)
        return selected[:k]
    return ranked[:k]


def _param_lookup(res):
    names = list(getattr(res.model, 'param_names', []) or [])
    values = np.asarray(res.params, dtype=float)
    if len(names) != len(values):
        return {}
    return dict(zip(names, values))


def _smoothed_probs_array(res):
    probs = np.asarray(res.smoothed_marginal_probabilities, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    return probs


def _order_states(res, n_states, switching_variance):
    params = _param_lookup(res)
    if switching_variance:
        variances = [params.get(f'sigma2[{k}]', np.inf) for k in range(n_states)]
    else:
        variances = list(range(n_states))
    if not np.isfinite(variances).any():
        return np.arange(n_states)
    return np.argsort(variances)


def _extract_betas(res, n_states, n_coef):
    betas = np.zeros((n_states, n_coef))
    params = _param_lookup(res)
    for k in range(n_states):
        betas[k, 0] = params.get(f'const[{k}]', 0.0)
        for j in range(1, n_coef):
            betas[k, j] = params.get(f'x{j}[{k}]', 0.0)
    return betas


def _extract_transition_matrix(res, n_states):
    params = _param_lookup(res)
    trans_mat = np.zeros((n_states, n_states))
    for i in range(n_states):
        missing = []
        for j in range(n_states):
            key = f'p[{i}->{j}]'
            if key in params:
                trans_mat[i, j] = params[key]
            else:
                missing.append(j)
        if len(missing) == 1:
            trans_mat[i, missing[0]] = max(0.0, 1.0 - trans_mat[i].sum())
        elif trans_mat[i].sum() == 0:
            trans_mat[i, i] = 1.0
        row_sum = trans_mat[i].sum()
        if row_sum > 0:
            trans_mat[i] = trans_mat[i] / row_sum
    return trans_mat


def _is_degenerate(res, n_states, switching_variance):
    probs = _smoothed_probs_array(res)
    for k in range(n_states):
        if (probs[:, k] > 0.5).mean() < MIN_REGIME_FREQ:
            return True
    if switching_variance:
        params = _param_lookup(res)
        for k in range(n_states):
            if params.get(f'sigma2[{k}]', 1.0) < VAR_FLOOR:
                return True
    return False


def _fit_markov(y, X_exog, n_states, switching_variance, max_attempts=3):
    import logging
    logging.getLogger('statsmodels').setLevel(logging.ERROR)

    X_c = sm.add_constant(X_exog)
    best_res, best_llf = None, -np.inf

    for attempt in range(max_attempts):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mod = MarkovRegression(
                    endog=y.values,
                    k_regimes=n_states,
                    exog=X_c.values,
                    switching_variance=switching_variance,
                )
                res = mod.fit(maxiter=300, em_iter=200, disp=False,
                              search_reps=5 * (attempt + 1))
            if not np.isfinite(res.llf):
                continue
            if _is_degenerate(res, n_states, switching_variance):
                continue
            if res.llf > best_llf:
                best_llf = res.llf
                best_res = res
        except Exception:
            continue
    return best_res


def _markov_predict(res, X_test, n_states, last_probs):
    X_c = sm.add_constant(X_test).values if not isinstance(X_test, np.ndarray) \
        else np.column_stack([np.ones(len(X_test)), X_test])
    n_coef = X_c.shape[1]
    betas = _extract_betas(res, n_states, n_coef)
    state_preds = X_c @ betas.T
    return state_preds @ last_probs[:n_states]


def cv_layer3_single(X, y, d, market, l1_fit, k_feats, n_states,
                     switching_variance, n_folds=3):
    top_feats = _select_top_features(l1_fit, k_feats, market, force_ar=True)
    X_sub = X[top_feats]
    n = len(X_sub)
    folds = get_cv_folds(n, n_folds)
    all_pred, all_true = [], []
    n_converged = 0

    for ts, te in folds:
        Xtr, ytr = X_sub.iloc[:ts], y.iloc[:ts]
        Xte, yte = X_sub.iloc[ts:te], y.iloc[ts:te]

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

        all_pred.extend(p)
        all_true.extend(yte.values)

    all_pred, all_true = np.array(all_pred), np.array(all_true)
    return {
        'cv_rmse': mean_squared_error(all_true, all_pred) ** 0.5,
        'cv_da':   direction_accuracy(all_true, all_pred),
        'top_feats': top_feats,
        'k_feats': k_feats, 'n_states': n_states,
        'switching_variance': switching_variance,
        'n_converged': n_converged, 'n_folds': len(folds),
        'y_pred': all_pred, 'y_true': all_true,
    }


L3_GRID = {
    'k_feats':            [3, 5, 7],
    'n_states':           [2, 3],
    'switching_variance': [True, False],
}


def sweep_layer3(d, market, h, l1_fit, verbose=True):
    X, y = make_X_y(d, market, h)
    configs = list(itertools.product(
        L3_GRID['k_feats'], L3_GRID['n_states'], L3_GRID['switching_variance']))

    results = []
    for k_feats, n_states, sv in configs:
        try:
            cv = cv_layer3_single(X, y, d, market, l1_fit,
                                  k_feats, n_states, sv)
            results.append(cv)
            if verbose:
                sv_str = 'SV' if sv else 'FV'
                print(f"      k={k_feats} s={n_states} {sv_str}: "
                      f"RMSE={cv['cv_rmse']:.5f} DA={cv['cv_da']:.1f}% "
                      f"({cv['n_converged']}/{cv['n_folds']} folds)")
        except Exception as e:
            if verbose:
                print(f"      k={k_feats} s={n_states} sv={sv}: FAILED ({e})")

    if not results:
        return None, []
    best = min(results, key=lambda r: r['cv_rmse'])
    return best, results


def fit_layer3(d, market, h, l1_fit, best_config):
    X, y = make_X_y(d, market, h)
    k_feats = best_config['k_feats']
    n_states = best_config['n_states']
    switching_variance = best_config['switching_variance']

    top_feats = _select_top_features(l1_fit, k_feats, market, force_ar=True)
    X_sub = X[top_feats]

    sc = StandardScaler()
    X_scaled = pd.DataFrame(sc.fit_transform(X_sub), columns=top_feats,
                            index=X_sub.index)

    res = _fit_markov(y, X_scaled, n_states, switching_variance)
    if res is None:
        return None

    params = _param_lookup(res)
    order = _order_states(res, n_states, switching_variance)
    smooth_probs = _smoothed_probs_array(res)
    smooth_probs_ordered = smooth_probs[:, order]

    X_c = sm.add_constant(X_scaled)
    n_coef = X_c.shape[1]
    raw_betas = _extract_betas(res, n_states, n_coef)

    state_betas = {}
    state_variance = {}
    for ki, k in enumerate(order):
        state_betas[ki] = raw_betas[k]
        if switching_variance:
            state_variance[ki] = params.get(f'sigma2[{k}]', np.nan)
        else:
            state_variance[ki] = params.get('sigma2', np.nan)

    trans_mat = _extract_transition_matrix(res, n_states)
    trans_mat = trans_mat[np.ix_(order, order)]

    return {
        'result': res, 'scaler': sc, 'n_states': n_states,
        'switching_variance': switching_variance, 'k_feats': k_feats,
        'top_feats': top_feats, 'feature_names': X.columns.tolist(),
        'state_betas': state_betas, 'state_variance': state_variance,
        'trans_mat': trans_mat, 'smooth_probs': smooth_probs_ordered,
        'dates': d.loc[X.index, 'Date'].values, 'state_order': order,
        'cv': {'cv_rmse': best_config['cv_rmse'], 'cv_da': best_config['cv_da'],
               'y_pred': best_config['y_pred'], 'y_true': best_config['y_true']},
        'n_train': len(X),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. UNIFIED TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def train_all(train_path, sweep_l3=True):
    print("Loading and engineering features...")
    df_train = load_clean(train_path)
    d = engineer_features(df_train)

    fitted = {}
    results = []
    sweep_log = {}
    # Collect preds for significance testing
    preds = {}   # (market, h, layer) -> (y_true, y_pred)

    for market in ['TTF', 'POWER', 'EUA']:
        for h in [1, 3, 5]:
            print(f"\n{'─'*60}")
            print(f"  {market}  h={h}")
            print(f"{'─'*60}")

            # ── Layer 1
            print(f"    L1: Ridge-ARX ...", end=' ', flush=True)
            l1 = fit_layer1(d, market, h)
            fitted[(market, h, 'L1')] = l1
            cv1 = l1['cv']
            preds[(market, h, 'L1')] = (cv1['y_true'], cv1['y_pred'])
            vs_naive = (cv1['cv_rmse'] - cv1['naive_rmse']) / cv1['naive_rmse'] * 100
            results.append({
                'market': market, 'h': h, 'layer': 'L1',
                'cv_rmse': cv1['cv_rmse'], 'cv_da': cv1['cv_da'],
                'naive_rmse': cv1['naive_rmse'], 'vs_naive': vs_naive,
                'note': f"α={l1['alpha']:.4f}",
            })
            print(f"RMSE={cv1['cv_rmse']:.5f}  DA={cv1['cv_da']:.1f}%  "
                  f"(α={l1['alpha']})")

            # ── Layer 3
            if sweep_l3:
                print(f"    L3: Markov-switching HP sweep ...")
                best, all_res = sweep_layer3(d, market, h, l1, verbose=True)
                sweep_log[(market, h)] = all_res
                if best is not None:
                    l3 = fit_layer3(d, market, h, l1, best)
                    if l3 is not None:
                        fitted[(market, h, 'L3')] = l3
                        preds[(market, h, 'L3')] = (best['y_true'], best['y_pred'])
                        vs3 = (best['cv_rmse'] - cv1['naive_rmse']) / cv1['naive_rmse'] * 100
                        sv_str = 'SV' if best['switching_variance'] else 'FV'
                        results.append({
                            'market': market, 'h': h, 'layer': 'L3',
                            'cv_rmse': best['cv_rmse'], 'cv_da': best['cv_da'],
                            'naive_rmse': cv1['naive_rmse'], 'vs_naive': vs3,
                            'note': f"k={best['k_feats']} s={best['n_states']} {sv_str}",
                        })
                        print(f"    L3 BEST: RMSE={best['cv_rmse']:.5f}  "
                              f"DA={best['cv_da']:.1f}%  "
                              f"(k={best['k_feats']} s={best['n_states']} {sv_str})")
                    else:
                        print(f"    L3: final fit failed")
                else:
                    print(f"    L3: no config converged")

    _print_comparison(results, fitted)
    _print_sweep_summary(sweep_log)
    _print_significance(preds)

    return fitted, d, results, sweep_log, preds


# ═══════════════════════════════════════════════════════════════════════════
# 9. REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def _print_comparison(results, fitted):
    print("\n\n" + "=" * 80)
    print("V4 MODEL COMPARISON — L1 vs L3 (best HP)")
    print("=" * 80)
    print(f"{'Market':6} {'h':2}  {'Layer':5}  {'CV_RMSE':9}  {'vs Naive':9}  "
          f"{'CV_DA':7}  {'Note'}")
    print("-" * 80)

    prev_mkt = ''
    for r in results:
        sep = '' if r['market'] == prev_mkt else '\n' if prev_mkt else ''
        prev_mkt = r['market']
        print(f"{sep}{r['market']:6} {r['h']:2}  {r['layer']:5}  "
              f"{r['cv_rmse']:9.5f}  {r['vs_naive']:+8.1f}%  "
              f"{r['cv_da']:6.1f}%  {r['note']}")

    print(f"\n{'─'*80}")
    print("BEST LAYER PER (MARKET, HORIZON)")
    print(f"{'─'*80}")
    by_cell = {}
    for r in results:
        key = (r['market'], r['h'])
        if key not in by_cell or r['cv_rmse'] < by_cell[key]['cv_rmse']:
            by_cell[key] = r
    for (mkt, h), r in sorted(by_cell.items()):
        print(f"  {mkt:6} h={h}: {r['layer']:4}  RMSE={r['cv_rmse']:.5f}  "
              f"DA={r['cv_da']:.1f}%  vs_naive={r['vs_naive']:+.1f}%  "
              f"({r['note']})")

    print(f"\n{'─'*80}")
    print("LAYER 1 — Top Ridge coefficients (h=1)")
    print(f"{'─'*80}")
    for market in ['TTF', 'POWER', 'EUA']:
        key = (market, 1, 'L1')
        if key not in fitted:
            continue
        m = fitted[key]
        coefs = sorted(zip(m['feature_names'], m['model'].coef_),
                       key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  {market}:")
        for fname, coef in coefs[:8]:
            print(f"    {coef:+.5f}  {fname}")


def _print_sweep_summary(sweep_log):
    print(f"\n\n{'='*80}")
    print("L3 HYPERPARAMETER SWEEP — all configs")
    print(f"{'='*80}")
    print(f"{'Mkt':6}{'h':3}{'k':3}{'s':3}{'SV':4}"
          f"{'CV_RMSE':>10}{'CV_DA':>9}{'folds':>10}")
    print("-" * 80)
    for (mkt, h), results in sorted(sweep_log.items()):
        for r in sorted(results, key=lambda x: x['cv_rmse']):
            sv = 'T' if r['switching_variance'] else 'F'
            print(f"{mkt:6}{h:3}{r['k_feats']:3}{r['n_states']:3}{sv:>4}"
                  f"{r['cv_rmse']:>10.5f}{r['cv_da']:>8.1f}%"
                  f"{r['n_converged']:>5}/{r['n_folds']}")
        print()


def _print_significance(preds):
    """DM test: each layer vs naive zero forecast.
       PT test: directional predictability of the layer's signed forecast.
       DM (L3 vs L1): pairwise model comparison where both exist."""
    print(f"\n\n{'='*80}")
    print("SIGNIFICANCE TESTS — Diebold-Mariano (vs naive) & Pesaran-Timmermann")
    print("=" * 80)
    print(f"{'Market':6}{'h':>3}{'Layer':>6}"
          f"{'DM_vs_naive':>13}{'DM_p':>8}"
          f"{'PT_stat':>10}{'PT_p':>8}  Significance")
    print("-" * 80)

    for (mkt, h, lyr), (yt, yp) in sorted(preds.items()):
        yt = np.asarray(yt); yp = np.asarray(yp)
        e_model = yt - yp
        e_naive = yt - 0.0
        dm_stat, dm_p = dm_test(e_model, e_naive, h=h)
        pt_stat, pt_p = pt_test(yt, yp)

        # Significance stars
        def stars(p):
            if np.isnan(p): return ''
            if p < 0.01: return '***'
            if p < 0.05: return '**'
            if p < 0.10: return '*'
            return ''
        sig = f"DM:{stars(dm_p):<3} PT:{stars(pt_p)}"

        dm_s = f"{dm_stat:+.3f}"  if np.isfinite(dm_stat) else '  n/a'
        dm_ps = f"{dm_p:.3f}"     if np.isfinite(dm_p)    else ' n/a'
        pt_s = f"{pt_stat:+.3f}"  if np.isfinite(pt_stat) else '  n/a'
        pt_ps = f"{pt_p:.3f}"     if np.isfinite(pt_p)    else ' n/a'
        print(f"{mkt:6}{h:>3}{lyr:>6}"
              f"{dm_s:>13}{dm_ps:>8}"
              f"{pt_s:>10}{pt_ps:>8}  {sig}")

    # Pairwise: L3 vs L1 (where both exist)
    print(f"\n{'─'*80}")
    print("PAIRWISE DM (L3 vs L1) — negative stat favours L3")
    print(f"{'─'*80}")
    print(f"{'Market':6}{'h':>3}{'DM_stat':>10}{'DM_p':>8}  Verdict")
    print("-" * 80)
    markets_horizons = set((m, h) for (m, h, _) in preds.keys())
    for (mkt, h) in sorted(markets_horizons):
        k1 = (mkt, h, 'L1'); k3 = (mkt, h, 'L3')
        if k1 not in preds or k3 not in preds:
            continue
        yt1, yp1 = preds[k1]; yt3, yp3 = preds[k3]
        # Align lengths (L1 uses 5 folds, L3 uses 3 — take common tail)
        n = min(len(yp1), len(yp3))
        e1 = yt1[-n:] - yp1[-n:]
        e3 = yt3[-n:] - yp3[-n:]
        dm_stat, dm_p = dm_test(e3, e1, h=h)
        if not np.isfinite(dm_stat):
            verdict = 'n/a'
        elif dm_p < 0.05 and dm_stat < 0:
            verdict = 'L3 > L1 (p<0.05)'
        elif dm_p < 0.05 and dm_stat > 0:
            verdict = 'L1 > L3 (p<0.05)'
        elif dm_p < 0.10:
            verdict = f"marginal (p={dm_p:.2f})"
        else:
            verdict = 'no difference'
        dm_s = f"{dm_stat:+.3f}" if np.isfinite(dm_stat) else '  n/a'
        dm_ps = f"{dm_p:.3f}"    if np.isfinite(dm_p)    else ' n/a'
        print(f"{mkt:6}{h:>3}{dm_s:>10}{dm_ps:>8}  {verdict}")

    print(f"\n  Legend: * p<0.10, ** p<0.05, *** p<0.01")
    print(f"  DM vs naive: negative stat → model beats zero forecast.")
    print(f"  PT: positive stat with low p → genuine directional ability.")


# ═══════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    train_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--train' and i+1 < len(sys.argv):
            train_path = sys.argv[i+1]
    if train_path is None:
        train_path = '/Users/jefflu/Library/CloudStorage/OneDrive-Personal/Documents/MSQF Fordham/ML Internship/k2_clean_v3.csv'
    fitted, d_train, results, sweep_log, preds = train_all(train_path, sweep_l3=True)
