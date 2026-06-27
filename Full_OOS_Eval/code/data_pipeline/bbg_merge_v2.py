"""
Merge Bloomberg Tier-2 pulls from KtE_BBG_Data_Extract-1.xlsx into k2_clean_v3.csv.

Adds two new feature families on top of v3:

  1. Vol surface (Option sheet) — 4-point IV (25D put / 25D call / 50D put / 50D call)
     for MO1 (EUA front) and TZT1 (TTF generic front). Other underlyings in the sheet
     only start 2024-05 (too short) or are empty (XA1) and are skipped.
     Derived per underlying: ATM_50D, RR_25D (skew), BF_25D (convexity),
     plus first differences and 20d z-scores of ATM and RR.

  2. BBG ESMA COT (COT sheet) on MO1 (EUA). Weekly Friday data with G1/G2/G3 split
     (Futures-only, Options-only, Futures+Options) that the existing MF_COT_EUA_*
     series in v3 lacks. Speculator proxy = "Investment Funds" (NC analog);
     hedger proxy = "Commercial Undertakings". F+O (G3) is the primary view.

Note: Prices/Volume/Open_Int sheets are *identical* to the original extract
(same tickers, same 2020-01-01 → 2026-04-17 window) so we do NOT re-run those
feature derivations here — they are already in v3 via bbg_merge.py.

Output: k2_clean_v4.csv alongside k2_clean_v3.csv.
"""

import numpy as np
import pandas as pd
from pathlib import Path

WORKSPACE = Path(__file__).parent
XLSX      = WORKSPACE / "KtE_BBG_Data_Extract-1.xlsx"
CLEAN_IN  = WORKSPACE / "KtE_Project" / "Jeff workspace" / "k2_clean_v3.csv"
CLEAN_OUT = WORKSPACE / "KtE_Project" / "Jeff workspace" / "k2_clean_v4.csv"

RK_MAP = {
    "RK910": "PUT_IV_25D",
    "RK903": "CALL_IV_25D",
    "RK908": "PUT_IV_50D",
    "RK901": "CALL_IV_50D",
}

USABLE_UNDERLYINGS = {
    "MO1":  "EUA",   # 746 non-null days from 2020-01
    "TZT1": "TTF",   # 802 non-null days from 2020-01
}


def _zscore(s: pd.Series, win: int = 20, min_p: int = 10) -> pd.Series:
    m = s.rolling(win, min_periods=min_p).mean()
    sd = s.rolling(win, min_periods=min_p).std(ddof=0)
    return (s - m) / sd


def load_option_surface() -> pd.DataFrame:
    """Return daily frame with columns: DATE + BBG_IV_{UL}_{RKFIELD} for usable underlyings."""
    raw = pd.read_excel(XLSX, sheet_name="Option", header=None)
    rk_codes  = raw.iloc[1].astype(str).tolist()
    underlyings = raw.iloc[3].astype(str).tolist()

    opt = pd.read_excel(XLSX, sheet_name="Option", header=3)
    opt = opt.rename(columns={opt.columns[0]: "DATE"})
    opt["DATE"] = pd.to_datetime(opt["DATE"], errors="coerce")
    opt = opt.dropna(subset=["DATE"])

    rename = {}
    for i, (col, rk_full, ul_full) in enumerate(zip(opt.columns, rk_codes, underlyings)):
        if col == "DATE":
            continue
        # rk_full looks like 'RK910MOA Comdty'; the leading 5 chars are the RK code
        rk = rk_full[:5] if isinstance(rk_full, str) and rk_full.startswith("RK") else None
        ul = ul_full.replace(" Comdty", "").strip() if isinstance(ul_full, str) else None
        if rk not in RK_MAP or ul not in USABLE_UNDERLYINGS:
            continue
        rename[col] = f"BBG_IV_{USABLE_UNDERLYINGS[ul]}_{RK_MAP[rk]}"

    opt = opt[["DATE"] + list(rename.keys())].rename(columns=rename)
    for c in opt.columns:
        if c != "DATE":
            opt[c] = pd.to_numeric(opt[c], errors="coerce")
    return opt


def derive_vol_features(iv: pd.DataFrame) -> pd.DataFrame:
    """Add ATM, RR, BF + d, z20 features for each underlying present."""
    out = iv.copy()
    for ul in set(USABLE_UNDERLYINGS.values()):
        p25 = f"BBG_IV_{ul}_PUT_IV_25D"
        c25 = f"BBG_IV_{ul}_CALL_IV_25D"
        p50 = f"BBG_IV_{ul}_PUT_IV_50D"
        c50 = f"BBG_IV_{ul}_CALL_IV_50D"
        if not all(c in out.columns for c in (p25, c25, p50, c50)):
            continue
        atm = (out[p50] + out[c50]) / 2.0
        rr  = out[c25] - out[p25]                  # negative = put-skew (tail-risk premium)
        bf  = (out[p25] + out[c25]) / 2.0 - atm    # convexity / wings richness
        out[f"BBG_IV_{ul}_ATM"] = atm
        out[f"BBG_IV_{ul}_RR25"] = rr
        out[f"BBG_IV_{ul}_BF25"] = bf
        out[f"D_BBG_IV_{ul}_ATM"] = atm.diff()
        out[f"D_BBG_IV_{ul}_RR25"] = rr.diff()
        out[f"BBG_IV_{ul}_ATM_Z20"] = _zscore(atm)
        out[f"BBG_IV_{ul}_RR25_Z20"] = _zscore(rr)
    return out


def load_cot_master() -> pd.DataFrame:
    ml = pd.read_excel(XLSX, sheet_name="Master List COT", header=1)
    ml = ml.rename(columns={"Bloomberg Ticker": "TICKER"})
    ml["TICKER"] = ml["TICKER"].astype(str).str.strip()
    return ml[["ESMA Category", "Direction", "Group", "Field", "TICKER"]].dropna(subset=["TICKER"])


def load_cot_values() -> pd.DataFrame:
    """Weekly COT — header on row 4 (0-indexed). Returns wide DataFrame keyed on DATE."""
    cot = pd.read_excel(XLSX, sheet_name="COT", header=4)
    cot = cot.rename(columns={cot.columns[0]: "DATE"})
    cot["DATE"] = pd.to_datetime(cot["DATE"], errors="coerce")
    cot = cot.dropna(subset=["DATE"])
    # Drop Excel-serial junk rows (e.g. 1900-01-07)
    cot = cot[cot["DATE"] >= "2010-01-01"].reset_index(drop=True)
    for c in cot.columns:
        if c != "DATE":
            cot[c] = pd.to_numeric(cot[c], errors="coerce")
    return cot


def derive_cot_features(master: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    """Pick the handful of COT series that matter and name them by role."""
    # Build lookup: (Category, Direction, Group, Field) -> ticker
    key = master.set_index(["ESMA Category", "Direction", "Group", "Field"])["TICKER"]

    def pick(cat, direction, group, field):
        try:
            t = key.loc[(cat, direction, group, field)]
        except KeyError:
            return None
        return t if t in values.columns else None

    # MO1 = EUA. Speculator = Investment Funds (NC analog); Hedger = Commercial Undertakings.
    specs = [
        ("BBG_COT_EUA_SPEC_NET_FO",       "Investment Funds",        "Net",   "G3", "Position"),
        ("BBG_COT_EUA_SPEC_LONG_FO",      "Investment Funds",        "Long",  "G3", "Position"),
        ("BBG_COT_EUA_SPEC_SHORT_FO",     "Investment Funds",        "Short", "G3", "Position"),
        ("BBG_COT_EUA_SPEC_LONG_PCTOI_FO","Investment Funds",        "Long",  "G3", "PctOI"),
        ("BBG_COT_EUA_SPEC_SHORT_PCTOI_FO","Investment Funds",       "Short", "G3", "PctOI"),
        ("BBG_COT_EUA_HEDGE_NET_FO",      "Commercial Undertakings", "Net",   "G3", "Position"),
        ("BBG_COT_EUA_HEDGE_LONG_FO",     "Commercial Undertakings", "Long",  "G3", "Position"),
        ("BBG_COT_EUA_HEDGE_SHORT_FO",    "Commercial Undertakings", "Short", "G3", "Position"),
        ("BBG_COT_EUA_IF_NET_FO",         "Investment Firms or Credit Institutions", "Net",  "G3", "Position"),
        ("BBG_COT_EUA_OFI_NET_FO",        "Other Financial Institutions",            "Net",  "G3", "Position"),
    ]
    out = values[["DATE"]].copy()
    for new_name, cat, direction, group, field in specs:
        t = pick(cat, direction, group, field)
        if t is None:
            print(f"  [skip] {new_name}: no ticker in Master List or missing from values")
            continue
        out[new_name] = values[t]

    # Derived: speculator net change, net as % of gross, z-scores
    if "BBG_COT_EUA_SPEC_NET_FO" in out.columns:
        out["D_BBG_COT_EUA_SPEC_NET_FO"]  = out["BBG_COT_EUA_SPEC_NET_FO"].diff()
        out["BBG_COT_EUA_SPEC_NET_Z52"]   = _zscore(out["BBG_COT_EUA_SPEC_NET_FO"], win=52, min_p=20)
    if {"BBG_COT_EUA_SPEC_LONG_FO", "BBG_COT_EUA_SPEC_SHORT_FO"}.issubset(out.columns):
        gross = out["BBG_COT_EUA_SPEC_LONG_FO"] + out["BBG_COT_EUA_SPEC_SHORT_FO"]
        out["BBG_COT_EUA_SPEC_NET_PCTGROSS"] = (
            (out["BBG_COT_EUA_SPEC_LONG_FO"] - out["BBG_COT_EUA_SPEC_SHORT_FO"]) / gross.where(gross > 0)
        )
    if "BBG_COT_EUA_HEDGE_NET_FO" in out.columns:
        out["D_BBG_COT_EUA_HEDGE_NET_FO"] = out["BBG_COT_EUA_HEDGE_NET_FO"].diff()
    return out


def main() -> None:
    clean = pd.read_csv(CLEAN_IN)
    clean["DATE"] = pd.to_datetime(clean["DATE"])
    print(f"k2_clean_v3: {len(clean)} rows, {len(clean.columns)} cols, "
          f"{clean.DATE.min().date()} → {clean.DATE.max().date()}")

    # --- Vol surface
    iv_raw = load_option_surface()
    iv_full = derive_vol_features(iv_raw)
    print(f"\nVol surface: {iv_full.shape[0]} rows, {iv_full.shape[1] - 1} cols")

    # --- COT
    cot_master = load_cot_master()
    cot_values = load_cot_values()
    cot_feat = derive_cot_features(cot_master, cot_values)
    print(f"COT (weekly): {cot_feat.shape[0]} rows, {cot_feat.shape[1] - 1} cols, "
          f"{cot_feat.DATE.min().date()} → {cot_feat.DATE.max().date()}")

    # --- Merge: vol is daily, COT is weekly → ffill COT up to 7 days
    merged = clean.merge(iv_full, on="DATE", how="left") \
                  .merge(cot_feat, on="DATE", how="left")

    iv_cols  = [c for c in iv_full.columns  if c != "DATE"]
    cot_cols = [c for c in cot_feat.columns if c != "DATE"]

    # Forward-fill within the merged frame. IV: gaps up to 3d (market convention);
    # COT: up to 7d since it is a weekly observation.
    merged[iv_cols]  = merged[iv_cols].ffill(limit=3)
    merged[cot_cols] = merged[cot_cols].ffill(limit=7)

    new_cols = iv_cols + cot_cols
    print(f"\nAdded {len(new_cols)} columns ({len(iv_cols)} IV + {len(cot_cols)} COT).")
    cov = merged[new_cols].notna().mean() * 100
    print("Coverage %:")
    for c, pct in cov.sort_values(ascending=False).items():
        print(f"  {c:42s}  {pct:5.1f}%")

    merged.to_csv(CLEAN_OUT, index=False)
    print(f"\nwrote {CLEAN_OUT}")
    print(f"  {len(merged)} rows, {len(merged.columns)} cols "
          f"(+{len(merged.columns) - len(clean.columns)} new)")


if __name__ == "__main__":
    main()
