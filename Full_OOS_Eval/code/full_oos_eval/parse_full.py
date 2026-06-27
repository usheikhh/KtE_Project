"""Parse the raw K2E monthly-delivery Excel (3-row merged header, 101 cols)
into the k2_clean-compatible schema, by POSITION. The column order in every
K2E delivery follows the same template, so positional mapping to the known
101 COLUMN_IDs (taken from the validated oos_jan2025_parsed.csv) is exact.

COT columns (indices 27-65, the 39 MF_COT_* fields) are forward-filled to
propagate the weekly Wednesday ESMA publication across the week, mirroring
the original oos_parser behaviour.
"""
import sys
import pandas as pd
import numpy as np

# 101 canonical column ids in delivery order (col 0 = Date)
PARSED_COLS = list(pd.read_csv('data/oos_jan2025_parsed.csv', nrows=0).columns)
assert len(PARSED_COLS) == 101, len(PARSED_COLS)

# COT columns to forward-fill = the contiguous MF_COT_* block
COT_IDX = [i for i, c in enumerate(PARSED_COLS) if c.startswith('MF_COT_')]


def parse_oos_file(path):
    # data starts on the 4th row (3-row header). header=None, skiprows=3.
    raw = pd.read_excel(path, sheet_name='Dataset', header=None, skiprows=3)
    raw = raw.iloc[:, :101]
    raw.columns = PARSED_COLS
    raw = raw.rename(columns={PARSED_COLS[0]: 'Date'})
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    raw = raw.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    for c in raw.columns:
        if c != 'Date':
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
    # forward-fill weekly COT block
    cot_cols = [PARSED_COLS[i] for i in COT_IDX]
    raw[cot_cols] = raw[cot_cols].ffill()
    return raw


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else 'data/Full Dataset.xlsx'
    out = sys.argv[2] if len(sys.argv) > 2 else 'data/full_parsed.csv'
    df = parse_oos_file(inp)
    df.to_csv(out, index=False)
    print(f'Parsed {len(df)} rows x {len(df.columns)} cols  ({df.Date.min().date()} -> {df.Date.max().date()})')
    print(f'COT cols ffilled: {len(COT_IDX)}')

    # ---- VALIDATION against the original Jan-2025 parse ----
    ref = pd.read_csv('data/oos_jan2025_parsed.csv', parse_dates=['Date'])
    sub = df[(df.Date >= ref.Date.min()) & (df.Date <= ref.Date.max())].reset_index(drop=True)
    print(f'\nVALIDATION vs oos_jan2025_parsed.csv: ref {len(ref)} rows, full-subset {len(sub)} rows')
    # align on Date
    m = ref.merge(sub, on='Date', suffixes=('_ref', '_new'))
    print(f'matched dates: {len(m)}')
    maxdiff = {}
    for c in PARSED_COLS[1:]:
        a, b = m[f'{c}_ref'], m[f'{c}_new']
        d = (a - b).abs()
        both_nan = a.isna() & b.isna()
        d = d[~both_nan]
        md = d.max() if len(d) else 0.0
        if pd.notna(md) and md > 1e-6:
            maxdiff[c] = md
    if not maxdiff:
        print('PERFECT MATCH: all 100 data columns identical to original parse.')
    else:
        print(f'{len(maxdiff)} cols differ:')
        for c, v in sorted(maxdiff.items(), key=lambda x: -x[1])[:20]:
            print(f'  {c}: maxabs diff {v:.6g}')
