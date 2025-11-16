# main_extreme_scenarios.py
# Seasonal bootstrap with automatic extreme/typical year classification
# Based on hydrological year total flows: 5%/10%/80%/90%/95% percentiles

import os
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import quote
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
WMIP_URL = "https://water-monitoring.information.qld.gov.au/cgi/webservice.pl"
SITE = os.environ.get("WMIP_SITE", "120002C")
START = os.environ.get("WMIP_START", "19681001000000")
END = os.environ.get("WMIP_END") or datetime.now().strftime("%Y%m%d000000")

DS_Q = os.environ.get("WMIP_DS_Q", "ATQ")
DS_L = os.environ.get("WMIP_DS_L", "AT")

# Add to Config section at top:
BOOTSTRAP_START_YEAR = int(os.environ.get("BOOT_BASELINE_START", "1990"))
BOOTSTRAP_END_YEAR = int(os.environ.get("BOOT_BASELINE_END", "2020"))

# Bootstrap knobs
BLOCK_DAYS = int(os.environ.get("BOOT_BLOCK_DAYS", "14"))
SEED = int(os.environ.get("BOOT_SEED", "42"))
TARGET_END_YEAR = int(os.environ.get("BOOT_TARGET_END_YEAR", "2099"))
MAX_BLOCK_REUSE_25Y = int(os.environ.get("BOOT_MAX_BLOCK_REUSE", "6"))
P_LONG_BLOCK = float(os.environ.get("BOOT_P_LONG", "0.70"))
LONG_BLOCK_SIZE = int(os.environ.get("BOOT_LONG_BLOCK", "90"))

# Seasonal definitions
WET_MONTHS = {11, 12, 1, 2, 3, 4}
DRY_MONTHS = {5, 6, 7, 8, 9, 10}

OUT_DIR = os.environ.get("WMIP_OUT_DIR", os.path.join(os.path.dirname(__file__), "wmipData"))


# -----------------------------
# WMIP fetch + parse
# -----------------------------
def _get_url(varcode: str, datasource: str) -> str:
    obj = {"function": "get_ts_traces", "version": "2", "params": {
        "site_list": SITE, "datasource": datasource, "varfrom": varcode, "varto": varcode,
        "start_time": START, "end_time": END, "data_type": "mean", "interval": "day", "multiplier": "1"}}
    return f"{WMIP_URL}?{quote(json.dumps(obj), safe='')}"


def fetch_raw(varcode: str, datasource: str) -> dict:
    r = requests.get(_get_url(varcode, datasource), timeout=180)
    r.raise_for_status()
    return r.json()


def extract_series(data: dict, value_name: str) -> pd.DataFrame:
    rows = []
    for ts in (data.get("return", {}).get("traces") or []):
        for pt in (ts.get("trace") or []):
            t, v, q = pt.get("t"), pt.get("v"), pt.get("q")
            if t is not None and v is not None:
                rows.append({"Date": t, value_name: v, f"{value_name}_quality": q})

    if not rows:
        for ts in (data.get("return_data") or []):
            for item in (ts.get("data") or []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    t, v = item[0], item[1]
                    q = item[2] if len(item) >= 3 else None
                    rows.append({"Date": t, value_name: v, f"{value_name}_quality": q})
                elif isinstance(item, dict):
                    t = item.get("t") or item.get("time")
                    v = item.get("v") or item.get("value")
                    q = item.get("q") or item.get("quality")
                    if t is not None and v is not None:
                        rows.append({"Date": t, value_name: v, f"{value_name}_quality": q})

    if not rows:
        return pd.DataFrame(columns=["Date", value_name, f"{value_name}_quality"])

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d%H%M%S", errors="coerce").dt.normalize()
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def consolidate_daily(df_q: pd.DataFrame, df_l: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(df_q, df_l, on="Date", how="outer").sort_values("Date").reset_index(drop=True)

    # Find the last date with good quality data
    if 'CUMECS_quality' in df.columns:
        quality = pd.to_numeric(df['CUMECS_quality'], errors='coerce')
        good_data_mask = (quality < 200) & (quality != 255)

        # Find last continuous stretch of good data
        # Allow small gaps (< 30 days) but stop at large continuous bad data
        df['good'] = good_data_mask
        df['bad_run'] = (~df['good']).rolling(window=30, min_periods=1).sum()

        # Find where we hit 30 consecutive bad quality codes
        continuous_bad = df['bad_run'] >= 30
        if continuous_bad.any():
            last_good_idx = continuous_bad.idxmax() - 30
            last_good_date = df.loc[last_good_idx, 'Date']
            print(f"\n  Detected end of quality data: {last_good_date.date()}")
            print(f"  Truncating {len(df) - last_good_idx} records with missing data")
            df = df.iloc[:last_good_idx + 1].copy()

        df = df.drop(columns=['good', 'bad_run'])

    # Calculate initial ML_day
    df["ML_day"] = pd.to_numeric(df.get("CUMECS"), errors='coerce') * 86.4

    start = df["Date"].min().normalize()
    end = df["Date"].max().normalize()
    full_idx = pd.date_range(start, end, freq="D")
    df = df.set_index("Date").reindex(full_idx).rename_axis("Date").reset_index()

    # Use quality codes to identify bad data, then interpolate
    for col in ["CUMECS", "Level_m"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")

            # Mark bad data based on quality codes
            quality_col = f"{col}_quality" if f"{col}_quality" in df.columns else None
            if quality_col and quality_col in df.columns:
                quality = df[quality_col]
                bad_quality = pd.to_numeric(quality, errors="coerce")
                bad_mask = (bad_quality >= 200) | (bad_quality == 255)
                if bad_mask.sum() > 0:
                    print(f"  {col}: marked {bad_mask.sum()} values as bad based on quality codes")
                    s[bad_mask] = np.nan

            # Remove physically impossible values
            negative_count = (s < 0).sum()
            if negative_count > 0:
                print(f"  {col}: removed {negative_count} negative values")
            s[s < 0] = np.nan

            # Simple linear interpolation for missing/bad values (limit to 30 days)
            null_before = s.isna().sum()
            s = s.interpolate(method='linear', limit=30, limit_direction='both')
            null_after = s.isna().sum()
            if null_before > null_after:
                print(f"  {col}: interpolated {null_before - null_after} missing values")

            df[col] = s

    # RECALCULATE ML_day from the cleaned/interpolated CUMECS values
    df["ML_day"] = pd.to_numeric(df["CUMECS"], errors="coerce") * 86.4
    print(f"  ML_day: recalculated from cleaned CUMECS values")

    # Remove leap days
    df = df[~((df["Date"].dt.month == 2) & (df["Date"].dt.day == 29))].reset_index(drop=True)
    return df[["Date", "ML_day", "CUMECS", "Level_m"]]

# -----------------------------
# Hydrological year classification
# -----------------------------
def hydro_year(date: pd.Timestamp) -> int:
    """Returns hydrological year (Nov-Oct). Nov/Dec belong to next calendar year."""
    if date.month >= 11:
        return date.year + 1
    else:
        return date.year


def classify_years_by_flow(base: pd.DataFrame, min_days: int = 350,
                           start_year: int = None, end_year: int = None) -> dict:
    """
    Classify hydrological years into 5 categories based on total annual flow.

    start_year/end_year: Optional - restrict classification to this period
    """
    b = base.copy()
    b["HydroYear"] = b["Date"].apply(hydro_year)

    # Filter to specified year range if provided
    if start_year or end_year:
        if start_year:
            b = b[b["HydroYear"] >= start_year]
        if end_year:
            b = b[b["HydroYear"] <= end_year]
        print(f"\nRestricting classification to hydro years {start_year or 'all'} to {end_year or 'all'}")

    # Calculate total flow and day count per hydrological year
    yearly_stats = b.groupby("HydroYear").agg(
        total_flow=("ML_day", "sum"),
        day_count=("Date", "count")
    ).dropna()

    # Filter to complete years only
    complete_years = yearly_stats[yearly_stats["day_count"] >= min_days]

    if len(complete_years) < 20:
        raise ValueError(
            f"Not enough complete hydrological years ({len(complete_years)}) in specified range for classification")

    excluded_years = set(yearly_stats.index) - set(complete_years.index)
    if excluded_years:
        print(f"\nExcluded incomplete hydrological years: {sorted(excluded_years)}")
        for hy in sorted(excluded_years):
            days = yearly_stats.loc[hy, "day_count"]
            print(f"  Year {hy}: only {days} days (need {min_days})")

    yearly_totals = complete_years["total_flow"]

    # Calculate percentile thresholds
    p05 = np.percentile(yearly_totals, 5)
    p10 = np.percentile(yearly_totals, 10)
    p90 = np.percentile(yearly_totals, 90)
    p95 = np.percentile(yearly_totals, 95)

    classification = {
        'SEVERE_DROUGHT': set(),
        'DROUGHT': set(),
        'TYPICAL': set(),
        'EXTREME_RAIN': set(),
        'SEVERE_RAIN': set()
    }

    for hy, total in yearly_totals.items():
        if total <= p05:
            classification['SEVERE_DROUGHT'].add(int(hy))
        elif total <= p10:
            classification['DROUGHT'].add(int(hy))
        elif total >= p95:
            classification['SEVERE_RAIN'].add(int(hy))
        elif total >= p90:
            classification['EXTREME_RAIN'].add(int(hy))
        else:
            classification['TYPICAL'].add(int(hy))

    print(f"\nYear classification by total flow ({len(complete_years)} complete years):")
    print(
        f"  SEVERE_DROUGHT ({len(classification['SEVERE_DROUGHT'])} years, <5th %ile): {sorted(classification['SEVERE_DROUGHT'])}")
    print(f"  DROUGHT ({len(classification['DROUGHT'])} years, 5th-10th %ile): {sorted(classification['DROUGHT'])}")
    print(f"  TYPICAL ({len(classification['TYPICAL'])} years, 10th-90th %ile): {sorted(classification['TYPICAL'])}")
    print(
        f"  EXTREME_RAIN ({len(classification['EXTREME_RAIN'])} years, 90th-95th %ile): {sorted(classification['EXTREME_RAIN'])}")
    print(
        f"  SEVERE_RAIN ({len(classification['SEVERE_RAIN'])} years, >95th %ile): {sorted(classification['SEVERE_RAIN'])}")
    print(f"\n  Percentiles:")
    print(f"    5th:  {p05:,.0f} ML/year")
    print(f"    10th: {p10:,.0f} ML/year")
    print(f"    90th: {p90:,.0f} ML/year")
    print(f"    95th: {p95:,.0f} ML/year")

    if excluded_years:
        print(f"\nNote: {len(excluded_years)} incomplete year(s) excluded from classification")

    return classification

# -----------------------------
# Seasonal helpers
# -----------------------------
def tag_season(d: pd.Timestamp) -> str:
    m = d.month
    if m in WET_MONTHS: return "WET"
    if m in DRY_MONTHS: return "DRY"
    return "UNK"

def build_season_pools(base: pd.DataFrame, year_set: set, include_incomplete: bool = True) -> dict:
    """
    Build seasonal pools from specified hydrological years.
    year_set: set of hydrological year numbers to include
    include_incomplete: if True, also includes data from incomplete years at boundaries
    """
    b = base.copy()
    b["Season"] = b["Date"].apply(tag_season)
    b["HydroYear"] = b["Date"].apply(hydro_year)

    if include_incomplete:
        # For TYPICAL bootstrap: use complete years for classification,
        # but allow sampling from ALL years including incomplete boundary years
        b_filtered = b[b["HydroYear"].isin(year_set)].reset_index(drop=True)

        # Also include any boundary years (incomplete) that weren't classified
        all_years = set(b["HydroYear"].unique())
        boundary_years = all_years - year_set
        if boundary_years:
            b_boundary = b[b["HydroYear"].isin(boundary_years)].reset_index(drop=True)
            b_filtered = pd.concat([b_filtered, b_boundary], ignore_index=True)
    else:
        # For extreme scenarios: only use the specified complete years
        b_filtered = b[b["HydroYear"].isin(year_set)].reset_index(drop=True)

    if b_filtered.empty:
        raise ValueError(f"No data for specified years: {year_set}")

    pools = {}
    for s in ["WET", "DRY"]:
        pool = b_filtered[b_filtered["Season"] == s].reset_index(drop=True)
        if pool.empty:
            raise ValueError(f"No data for season {s} in specified years.")
        pools[s] = pool

    return pools


# -----------------------------
# Block weights
# -----------------------------
def compute_block_weights(pool: pd.DataFrame, block_days: int = 14) -> np.ndarray:
    """
    Light downweighting of extreme events (99.5th percentile and above).
    """
    n = len(pool)
    ml = pd.to_numeric(pool["ML_day"].values, errors="coerce")
    valid = ml[~np.isnan(ml)]

    if valid.size == 0:
        return np.ones(n) / n

    # Target 99.5th percentile instead of 98th - much more selective
    p995 = np.percentile(valid, 99)
    flags = (ml > p995).astype(float)

    # Calculate fraction of extreme days in each block
    ext_frac = np.zeros(n, dtype=float)
    for i in range(n):
        idxs = [(i + k) % n for k in range(block_days)]
        ext_frac[i] = np.nanmean(flags[idxs])

    # Gentler weighting - only suppress blocks with extreme content
    # Changed from 0.12 to 0.3 for lighter touch
    w = 1.0 / (0.3 + ext_frac)
    w[np.isnan(w)] = 1.0
    w = np.maximum(w, 1e-6)
    return w / w.sum()

# -----------------------------
# Seasonal bootstrap
# -----------------------------
def seasonal_block_bootstrap(
        base: pd.DataFrame,
        year_set: set,
        start_year: int,
        end_year: int,
        block_size: int = 14,
        p_long_block: float = 0.50,
        long_block_size: int = 60,
        max_block_reuse_25y: int = 6,
        seed: int = 42
) -> pd.DataFrame:
    """
    Bootstrap from specified hydrological years.
    year_set: set of hydrological years to sample from
    """
    rng = np.random.default_rng(seed)
    pools = build_season_pools(base, year_set, include_incomplete=True)
    weights = {s: compute_block_weights(p, block_size) for s, p in pools.items()}
    reuse_counts = {s: np.zeros(len(pools[s]), dtype=int) for s in pools}

    sy_rows = []
    for year in range(start_year, end_year + 1):
        months = [(m, "WET" if m in WET_MONTHS else "DRY") for m in range(1, 13)]
        y_rows = []

        for month, season in months:
            pool = pools[season]
            n = len(pool)
            w_base = weights[season]

            if (year - start_year) % 25 == 0 and year != start_year:
                reuse_counts[season][:] = 0

            days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
            idxs = []

            while len(idxs) < days_in_month:
                capped = reuse_counts[season] >= max_block_reuse_25y
                w_eff = np.where(capped, 0.0, w_base)
                if w_eff.sum() == 0:
                    reuse_counts[season][:] = 0
                    w_eff = w_base
                w_eff = w_eff / w_eff.sum()

                start_idx = int(rng.choice(np.arange(n), p=w_eff))
                this_len = long_block_size if rng.random() < p_long_block else block_size
                block = [(start_idx + k) % n for k in range(this_len)]
                idxs.extend(block)
                reuse_counts[season][start_idx] += 1

            idxs = idxs[:days_in_month]
            sel = pool.iloc[idxs][["ML_day", "CUMECS", "Level_m"]].reset_index(drop=True)
            dates = pd.date_range(f"{year}-{month:02d}-01", periods=days_in_month, freq="D")
            sel.insert(0, "Date", dates)
            sel = sel[~((sel["Date"].dt.month == 2) & (sel["Date"].dt.day == 29))].reset_index(drop=True)
            y_rows.append(sel)

        ydf = pd.concat(y_rows, ignore_index=True)
        sy_rows.append(ydf)

    return pd.concat(sy_rows, ignore_index=True)


# -----------------------------
# Build extreme scenario (hydrological year)
# -----------------------------
def build_extreme_scenario(base: pd.DataFrame, year_set: set, scenario_name: str) -> pd.DataFrame:
    """
    Creates a 365-day canonical hydrological year (Nov 1 - Oct 31) from the most extreme year.
    Returns data with template dates (2000-11-01 to 2001-10-31).
    """
    b = base.copy()
    b["HydroYear"] = b["Date"].apply(hydro_year)

    # Calculate totals for each year in the set
    year_totals = []
    for hy in year_set:
        start_date = pd.Timestamp(f"{hy - 1}-11-01")
        end_date = pd.Timestamp(f"{hy}-10-31")

        year_data = b[(b["Date"] >= start_date) & (b["Date"] <= end_date)].copy()
        year_data = year_data[~((year_data["Date"].dt.month == 2) &
                                (year_data["Date"].dt.day == 29))].reset_index(drop=True)

        if len(year_data) >= 350:
            total_flow = year_data["ML_day"].sum()
            year_totals.append((hy, year_data, total_flow))

    if not year_totals:
        raise ValueError(f"No complete hydrological years for {scenario_name}")

    # Select most extreme
    if "RAIN" in scenario_name:
        best_hy, best_data, best_flow = max(year_totals, key=lambda x: x[2])
    else:  # DROUGHT
        best_hy, best_data, best_flow = min(year_totals, key=lambda x: x[2])

    result = best_data[["ML_day", "CUMECS", "Level_m"]].reset_index(drop=True)

    # Ensure exactly 365 days
    if len(result) > 365:
        result = result.iloc[:365]
    elif len(result) < 365:
        while len(result) < 365:
            result = pd.concat([result, result.iloc[[-1]]], ignore_index=True)

    # Template dates (Nov 1, 2000 - Oct 31, 2001)
    template_dates = pd.date_range("2000-11-01", periods=365, freq="D")
    template_dates = template_dates[~((template_dates.month == 2) & (template_dates.day == 29))]

    result = result.iloc[:len(template_dates)].copy()
    result.insert(0, "Date", template_dates[:len(result)])

    print(f"  Selected hydro year {best_hy} for {scenario_name} (total: {best_flow:,.0f} ML)")

    return result


# -----------------------------
# FDC computation
# -----------------------------
def compute_fdc(df: pd.DataFrame, col: str = "ML_day",
                probs=(0.99, 0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01)) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    out = {"prob": [], "quantile": []}
    for p in probs:
        out["prob"].append(p)
        out["quantile"].append(float(np.quantile(s, 1 - p)))
    return pd.DataFrame(out)


def save_fdc_plot(baseline: pd.DataFrame, scenarios: dict,
                  out_dir: str, fname: str = "fdc_all_scenarios.png"):
    plt.figure(figsize=(12, 7))

    fb = compute_fdc(baseline, "ML_day")
    plt.plot(fb["prob"], fb["quantile"], label="Baseline (all years)",
             linewidth=2.5, color='black', linestyle='-')

    colors = {
        "TYPICAL": "blue",
        "DROUGHT": "orange",
        "SEVERE_DROUGHT": "red",
        "EXTREME_RAIN": "lightgreen",
        "SEVERE_RAIN": "darkgreen"
    }

    for name, df in scenarios.items():
        fs = compute_fdc(df, "ML_day")
        plt.plot(fs["prob"], fs["quantile"], label=name, linewidth=1.5,
                 color=colors.get(name, "gray"), linestyle='--', alpha=0.8)

    plt.gca().invert_xaxis()
    plt.xlabel("Exceedance probability")
    plt.ylabel("ML/day")
    plt.title("Flow Duration Curves – All Scenarios")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    out_path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"WMIP fetch: site={SITE} window={START}-{END}")

    # Fetch data
    print("\nFetching discharge data...")
    raw_q = fetch_raw("140.00", DS_Q)
    df_q = extract_series(raw_q, "CUMECS")
    if df_q.empty and DS_Q.upper() == "ATQ":
        print("ATQ returned no discharge. Trying AT…")
        raw_q = fetch_raw("140.00", "AT")
        df_q = extract_series(raw_q, "CUMECS")

    print(f"Fetching level data...")
    raw_l = fetch_raw("100.00", DS_L)
    df_l = extract_series(raw_l, "Level_m")
    if df_l.empty and DS_L.upper() != "AT":
        print("Preferred datasource returned no level. Trying AT…")
        raw_l = fetch_raw("100.00", "AT")
        df_l = extract_series(raw_l, "Level_m")

    with open(os.path.join(OUT_DIR, "discharge_140.json"), "w", encoding="utf-8") as f:
        json.dump(raw_q, f)
    with open(os.path.join(OUT_DIR, "level_100.json"), "w", encoding="utf-8") as f:
        json.dump(raw_l, f)

    print(f"\nExtracted {len(df_q)} discharge records")
    print(f"Extracted {len(df_l)} level records")

    # Diagnostic: Check quality codes
    if 'CUMECS_quality' in df_q.columns:
        print(f"\nQuality code distribution (discharge):")
        qc_counts = df_q['CUMECS_quality'].value_counts().sort_index()
        for qc, count in list(qc_counts.items())[:10]:
            print(f"  Code {qc}: {count:,} records")
        bad_count = ((pd.to_numeric(df_q['CUMECS_quality'], errors='coerce') >= 200) |
                     (pd.to_numeric(df_q['CUMECS_quality'], errors='coerce') == 255)).sum()
        print(f"  Records to be interpolated (quality >= 200 or 255): {bad_count:,}")

    if 'Level_m_quality' in df_l.columns:
        print(f"\nQuality code distribution (level):")
        qc_counts = df_l['Level_m_quality'].value_counts().sort_index()
        for qc, count in list(qc_counts.items())[:10]:
            print(f"  Code {qc}: {count:,} records")

    print("\nConsolidating daily data with quality control...")
    base = consolidate_daily(df_q, df_l)

    # Post-consolidation diagnostics
    print(f"\n{'=' * 60}")
    print("DATA QUALITY SUMMARY")
    print('=' * 60)

    full_range_days = (base['Date'].max() - base['Date'].min()).days + 1
    actual_days = len(base)
    expected_leap_days = len([d for d in pd.date_range(base['Date'].min(), base['Date'].max(), freq='D')
                              if d.month == 2 and d.day == 29])
    expected_days = full_range_days - expected_leap_days

    print(f"\nBaseline: {base['Date'].min().date()} → {base['Date'].max().date()}")
    print(f"  Full date range: {full_range_days:,} days")
    print(f"  Leap days removed: {expected_leap_days}")
    print(f"  Expected days: {expected_days:,}")
    print(f"  Actual days: {actual_days:,}")

    if actual_days == expected_days:
        print(f"  Status: ✓ Data is continuous")
    else:
        missing = expected_days - actual_days
        print(f"  Status: ✗ Missing {missing:,} days")
        all_dates = pd.date_range(base['Date'].min(), base['Date'].max(), freq='D')
        all_dates = all_dates[~((all_dates.month == 2) & (all_dates.day == 29))]
        missing_dates = all_dates.difference(base['Date'])
        if len(missing_dates) > 0:
            print(f"  First missing dates: {[d.date() for d in missing_dates[:10]]}")

    # Classify years by flow - using recent baseline period
    year_classes = classify_years_by_flow(base,
                                          start_year=BOOTSTRAP_START_YEAR,
                                          end_year=BOOTSTRAP_END_YEAR)

    # ========================================
    # Generate TYPICAL scenario
    # ========================================
    print(f"\n{'=' * 60}")
    print(f"GENERATING TYPICAL SCENARIO")
    print('=' * 60)

    last_obs = base["Date"].max()
    current_year = last_obs.year
    year_end = pd.Timestamp(f"{current_year}-12-31")

    # Check if we need to fill the remainder of the current year
    if last_obs < year_end:
        days_to_fill = (year_end - last_obs).days
        print(f"\nFilling partial year: {days_to_fill} days from {last_obs.date()} to {year_end.date()}")

        # Generate synthetic data for the partial year
        partial_year = seasonal_block_bootstrap(
            base,
            year_set=year_classes['TYPICAL'],
            start_year=current_year,
            end_year=current_year,
            block_size=BLOCK_DAYS,
            p_long_block=P_LONG_BLOCK,
            long_block_size=LONG_BLOCK_SIZE,
            max_block_reuse_25y=MAX_BLOCK_REUSE_25Y,
            seed=SEED
        )

        # Only keep days after last observation
        partial_year = partial_year[partial_year["Date"] > last_obs].copy()
        print(f"Generated {len(partial_year)} days for partial year")
    else:
        partial_year = pd.DataFrame(columns=["Date", "ML_day", "CUMECS", "Level_m"])
        print(f"\nCurrent year is complete, no partial year fill needed")

    # Generate complete future years
    if current_year < TARGET_END_YEAR:
        print(f"\nGenerating complete future years: {current_year + 1} to {TARGET_END_YEAR}")
        synth_future = seasonal_block_bootstrap(
            base,
            year_set=year_classes['TYPICAL'],
            start_year=current_year + 1,
            end_year=TARGET_END_YEAR,
            block_size=BLOCK_DAYS,
            p_long_block=P_LONG_BLOCK,
            long_block_size=LONG_BLOCK_SIZE,
            max_block_reuse_25y=MAX_BLOCK_REUSE_25Y,
            seed=SEED
        )
        print(f"Generated {len(synth_future)} days for future years")

        # Combine all synthetic data
        synth_typical = pd.concat([partial_year, synth_future], ignore_index=True)
    else:
        synth_typical = partial_year

    synth_typical = synth_typical.sort_values("Date").reset_index(drop=True)
    print(
        f"Total synthetic TYPICAL: {synth_typical['Date'].min().date()} → {synth_typical['Date'].max().date()} ({len(synth_typical):,} days)")

    # ========================================
    # Generate DROUGHT scenario (10th percentile)
    # ========================================
    print(f"\n{'=' * 60}")
    print("GENERATING DROUGHT SCENARIO (10th percentile)")
    print('=' * 60)

    drought_scenario = build_extreme_scenario(base, year_classes['DROUGHT'], "DROUGHT")

    # ========================================
    # Generate SEVERE DROUGHT scenario (5th percentile)
    # ========================================
    print(f"\n{'=' * 60}")
    print("GENERATING SEVERE DROUGHT SCENARIO (5th percentile)")
    print('=' * 60)

    severe_drought_scenario = build_extreme_scenario(base, year_classes['SEVERE_DROUGHT'], "SEVERE_DROUGHT")

    # ========================================
    # Generate EXTREME_RAIN scenario (90th percentile)
    # ========================================
    print(f"\n{'=' * 60}")
    print("GENERATING EXTREME RAIN SCENARIO (90th percentile)")
    print('=' * 60)

    rain_scenario = build_extreme_scenario(base, year_classes['EXTREME_RAIN'], "EXTREME_RAIN")

    # ========================================
    # Generate SEVERE RAIN scenario (95th percentile)
    # ========================================
    print(f"\n{'=' * 60}")
    print("GENERATING SEVERE RAIN SCENARIO (95th percentile)")
    print('=' * 60)

    severe_rain_scenario = build_extreme_scenario(base, year_classes['SEVERE_RAIN'], "SEVERE_RAIN")

    # ========================================
    # Save outputs
    # ========================================
    print(f"\n{'=' * 60}")
    print("SAVING OUTPUTS")
    print('=' * 60)

    # Save TYPICAL
    typical_combined = pd.concat([base, synth_typical],
                                 ignore_index=True).sort_values("Date").reset_index(drop=True)

    typical_path = os.path.join(OUT_DIR, "flows_bootstrap_typical.parquet")
    typical_combined.to_parquet(typical_path, index=False)
    print(f"Saved: {typical_path}")

    # Save DROUGHT (10th percentile)
    drought_path = os.path.join(OUT_DIR, "flows_extreme_drought.parquet")
    drought_scenario.to_parquet(drought_path, index=False)
    print(f"Saved: {drought_path}")

    # Save SEVERE DROUGHT (5th percentile)
    severe_drought_path = os.path.join(OUT_DIR, "flows_severe_drought.parquet")
    severe_drought_scenario.to_parquet(severe_drought_path, index=False)
    print(f"Saved: {severe_drought_path}")

    # Save EXTREME_RAIN (90th percentile)
    rain_path = os.path.join(OUT_DIR, "flows_extreme_rain.parquet")
    rain_scenario.to_parquet(rain_path, index=False)
    print(f"Saved: {rain_path}")

    # Save SEVERE RAIN (95th percentile)
    severe_rain_path = os.path.join(OUT_DIR, "flows_severe_rain.parquet")
    severe_rain_scenario.to_parquet(severe_rain_path, index=False)
    print(f"Saved: {severe_rain_path}")

    # Save metadata
    # In main(), update the metadata section:

    meta = {
        "title": "Sellheim daily flows – scenarios for reservoir modeling",
        "site": SITE,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline": {
            "start": str(base['Date'].min().date()),
            "end": str(base['Date'].max().date()),
            "days": len(base),
            "years": f"{base['Date'].min().year}-{base['Date'].max().year}",
            "data_source": {
                "discharge": DS_Q,
                "level": DS_L
            },
            "quality_control": {
                "bad_quality_codes_filtered": "≥200 or 255",
                "interpolation_method": "linear",
                "interpolation_limit_days": 30,
                "leap_days_removed": True
            }
        },
        "year_classification": {
            "method": "Hydrological year (Nov-Oct) total flow percentiles",
            "baseline_period": f"{BOOTSTRAP_START_YEAR}-{BOOTSTRAP_END_YEAR}",
            "classification_basis": "Annual total flow (ML/year)",
            "thresholds": {
                "5th_percentile": f"{np.percentile([base[base['Date'].apply(hydro_year).isin(year_classes['SEVERE_DROUGHT'] | year_classes['DROUGHT'] | year_classes['TYPICAL'] | year_classes['EXTREME_RAIN'] | year_classes['SEVERE_RAIN'])].groupby(base['Date'].apply(hydro_year))['ML_day'].sum().values], 5):,.0f} ML/year",
                "10th_percentile": f"{np.percentile([base[base['Date'].apply(hydro_year).isin(year_classes['SEVERE_DROUGHT'] | year_classes['DROUGHT'] | year_classes['TYPICAL'] | year_classes['EXTREME_RAIN'] | year_classes['SEVERE_RAIN'])].groupby(base['Date'].apply(hydro_year))['ML_day'].sum().values], 10):,.0f} ML/year",
                "90th_percentile": f"{np.percentile([base[base['Date'].apply(hydro_year).isin(year_classes['SEVERE_DROUGHT'] | year_classes['DROUGHT'] | year_classes['TYPICAL'] | year_classes['EXTREME_RAIN'] | year_classes['SEVERE_RAIN'])].groupby(base['Date'].apply(hydro_year))['ML_day'].sum().values], 90):,.0f} ML/year",
                "95th_percentile": f"{np.percentile([base[base['Date'].apply(hydro_year).isin(year_classes['SEVERE_DROUGHT'] | year_classes['DROUGHT'] | year_classes['TYPICAL'] | year_classes['EXTREME_RAIN'] | year_classes['SEVERE_RAIN'])].groupby(base['Date'].apply(hydro_year))['ML_day'].sum().values], 95):,.0f} ML/year"
            },
            "severe_drought_years": sorted(list(year_classes['SEVERE_DROUGHT'])),
            "drought_years": sorted(list(year_classes['DROUGHT'])),
            "typical_years": sorted(list(year_classes['TYPICAL'])),
            "extreme_rain_years": sorted(list(year_classes['EXTREME_RAIN'])),
            "severe_rain_years": sorted(list(year_classes['SEVERE_RAIN'])),
            "count": {
                "severe_drought": len(year_classes['SEVERE_DROUGHT']),
                "drought": len(year_classes['DROUGHT']),
                "typical": len(year_classes['TYPICAL']),
                "extreme_rain": len(year_classes['EXTREME_RAIN']),
                "severe_rain": len(year_classes['SEVERE_RAIN'])
            }
        },
        "bootstrap_parameters": {
            "block_size_days": BLOCK_DAYS,
            "long_block_size_days": LONG_BLOCK_SIZE,
            "long_block_probability": P_LONG_BLOCK,
            "max_block_reuse_per_25y": MAX_BLOCK_REUSE_25Y,
            "random_seed": SEED,
            "seasonal_definitions": {
                "wet_season": "Nov-Apr",
                "dry_season": "May-Oct"
            },
            "weighting": "Downweighted blocks containing 98th percentile flows"
        },
        "scenarios": {
            "TYPICAL": {
                "description": "Bootstrap from 10th-90th percentile years",
                "file": "flows_bootstrap_typical.parquet",
                "method": "Seasonal block bootstrap",
                "sampling_pool": f"{len(year_classes['TYPICAL'])} hydrological years",
                "start": str(typical_combined['Date'].min().date()),
                "end": str(typical_combined['Date'].max().date()),
                "days": len(typical_combined),
                "statistics": {
                    "mean_ML_day": f"{synth_typical['ML_day'].mean():.0f}",
                    "max_ML_day": f"{synth_typical['ML_day'].max():,.0f}",
                    "min_ML_day": f"{synth_typical['ML_day'].min():.1f}"
                }
            },
            "DROUGHT": {
                "description": "Moderate drought (10th percentile)",
                "file": "flows_extreme_drought.parquet",
                "method": "Single worst year from 10th percentile category",
                "template_span": "2000-11-01 to 2001-10-31",
                "days": 365,
                "source_year": sorted(list(year_classes['DROUGHT']))[0] if year_classes['DROUGHT'] else None,
                "statistics": {
                    "total_annual_ML": f"{drought_scenario['ML_day'].sum():,.0f}",
                    "mean_ML_day": f"{drought_scenario['ML_day'].mean():.0f}",
                    "max_ML_day": f"{drought_scenario['ML_day'].max():,.0f}"
                }
            },
            "SEVERE_DROUGHT": {
                "description": "Severe drought (5th percentile)",
                "file": "flows_severe_drought.parquet",
                "method": "Single worst year from 5th percentile category",
                "template_span": "2000-11-01 to 2001-10-31",
                "days": 365,
                "source_year": sorted(list(year_classes['SEVERE_DROUGHT']))[0] if year_classes[
                    'SEVERE_DROUGHT'] else None,
                "statistics": {
                    "total_annual_ML": f"{severe_drought_scenario['ML_day'].sum():,.0f}",
                    "mean_ML_day": f"{severe_drought_scenario['ML_day'].mean():.0f}",
                    "max_ML_day": f"{severe_drought_scenario['ML_day'].max():,.0f}"
                }
            },
            "EXTREME_RAIN": {
                "description": "Extreme rain (90th percentile)",
                "file": "flows_extreme_rain.parquet",
                "method": "Single wettest year from 90th percentile category",
                "template_span": "2000-11-01 to 2001-10-31",
                "days": 365,
                "source_year": sorted(list(year_classes['EXTREME_RAIN']))[-1] if year_classes['EXTREME_RAIN'] else None,
                "statistics": {
                    "total_annual_ML": f"{rain_scenario['ML_day'].sum():,.0f}",
                    "mean_ML_day": f"{rain_scenario['ML_day'].mean():.0f}",
                    "max_ML_day": f"{rain_scenario['ML_day'].max():,.0f}"
                }
            },
            "SEVERE_RAIN": {
                "description": "Severe rain/flooding (95th percentile)",
                "file": "flows_severe_rain.parquet",
                "method": "Single wettest year from 95th percentile category",
                "template_span": "2000-11-01 to 2001-10-31",
                "days": 365,
                "source_year": sorted(list(year_classes['SEVERE_RAIN']))[-1] if year_classes['SEVERE_RAIN'] else None,
                "statistics": {
                    "total_annual_ML": f"{severe_rain_scenario['ML_day'].sum():,.0f}",
                    "mean_ML_day": f"{severe_rain_scenario['ML_day'].mean():.0f}",
                    "max_ML_day": f"{severe_rain_scenario['ML_day'].max():,.0f}"
                }
            }
        },
        "usage_notes": {
            "typical_scenario": "Use for baseline/reference modeling. Contains full historical record plus bootstrap projection to 2099.",
            "extreme_templates": "365-day templates for drought/rain years. Use insert_extreme_year.py to inject into specific years.",
            "hydrological_year": "Nov 1 to Oct 31. Template dates are 2000-11-01 to 2001-10-31 for compatibility.",
            "no_leap_days": "Feb 29 removed from all scenarios for consistent 365-day years."
        },
        "citations": {
            "data_source": "Queensland Government Water Monitoring Information Portal (WMIP)",
            "url": "https://water-monitoring.information.qld.gov.au/",
            "method_reference": "Seasonal block bootstrap with hydrological year classification"
        }
    }

    meta_path = os.path.join(OUT_DIR, "flows_scenarios_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {meta_path}")

    # ========================================
    # Validation
    # ========================================
    print(f"\n{'=' * 60}")
    print("FLOW STATISTICS")
    print('=' * 60)

    # Create typical-only baseline for comparison
    base_typical = base.copy()
    base_typical["HydroYear"] = base_typical["Date"].apply(hydro_year)
    base_typical_only = base_typical[base_typical["HydroYear"].isin(year_classes['TYPICAL'])].copy()

    print(f"\nTYPICAL scenario:")
    print(f"  Peak flow: {synth_typical['ML_day'].max():,.0f} ML/day")
    print(f"  Mean flow: {synth_typical['ML_day'].mean():.0f} ML/day")
    print(f"  Min flow:  {synth_typical['ML_day'].min():.1f} ML/day")

    print(f"\nTypical baseline (for comparison):")
    print(f"  Peak flow: {base_typical_only['ML_day'].max():,.0f} ML/day")
    print(f"  Mean flow: {base_typical_only['ML_day'].mean():.0f} ML/day")
    print(f"  Min flow:  {base_typical_only['ML_day'].min():.1f} ML/day")

    print(f"\nDROUGHT scenario (10th percentile):")
    print(f"  Peak flow: {drought_scenario['ML_day'].max():,.0f} ML/day")
    print(f"  Mean flow: {drought_scenario['ML_day'].mean():.0f} ML/day")
    print(f"  Total annual: {drought_scenario['ML_day'].sum():,.0f} ML")

    print(f"\nSEVERE DROUGHT scenario (5th percentile):")
    print(f"  Peak flow: {severe_drought_scenario['ML_day'].max():,.0f} ML/day")
    print(f"  Mean flow: {severe_drought_scenario['ML_day'].mean():.0f} ML/day")
    print(f"  Total annual: {severe_drought_scenario['ML_day'].sum():,.0f} ML")

    print(f"\nEXTREME RAIN scenario (90th percentile):")
    print(f"  Peak flow: {rain_scenario['ML_day'].max():,.0f} ML/day")
    print(f"  Mean flow: {rain_scenario['ML_day'].mean():.0f} ML/day")
    print(f"  Total annual: {rain_scenario['ML_day'].sum():,.0f} ML")

    print(f"\nSEVERE RAIN scenario (95th percentile):")
    print(f"  Peak flow: {severe_rain_scenario['ML_day'].max():,.0f} ML/day")
    print(f"  Mean flow: {severe_rain_scenario['ML_day'].mean():.0f} ML/day")
    print(f"  Total annual: {severe_rain_scenario['ML_day'].sum():,.0f} ML")

    # FDC plots
    scenarios_dict = {
        "TYPICAL": synth_typical,
        "DROUGHT": drought_scenario,
        "SEVERE_DROUGHT": severe_drought_scenario,
        "EXTREME_RAIN": rain_scenario,
        "SEVERE_RAIN": severe_rain_scenario
    }
    fdc_path = save_fdc_plot(base, scenarios_dict, OUT_DIR)
    print(f"\nSaved FDC plot: {fdc_path}")

    # FDC table
    fdc_data = []
    for name, df in [("baseline_all", base), ("baseline_typical_only", base_typical_only),
                     ("TYPICAL", synth_typical),
                     ("DROUGHT", drought_scenario),
                     ("SEVERE_DROUGHT", severe_drought_scenario),
                     ("EXTREME_RAIN", rain_scenario),
                     ("SEVERE_RAIN", severe_rain_scenario)]:
        fdc = compute_fdc(df, "ML_day")
        for _, row in fdc.iterrows():
            fdc_data.append({
                "scenario": name,
                "exceedance_prob": row["prob"],
                "flow_ML_day": row["quantile"]
            })

    # Add after the FLOW STATISTICS section in main():

    print(f"\n{'=' * 60}")
    print("EXTREME EVENT FREQUENCY VALIDATION")
    print('=' * 60)

    # Define thresholds based on typical year data
    base_typical = base.copy()
    base_typical["HydroYear"] = base_typical["Date"].apply(hydro_year)
    base_typical_only = base_typical[base_typical["HydroYear"].isin(year_classes['TYPICAL'])].copy()

    # Calculate percentiles from typical years only
    p95 = np.percentile(base_typical_only['ML_day'], 95)
    p99 = np.percentile(base_typical_only['ML_day'], 99)
    p999 = np.percentile(base_typical_only['ML_day'], 99.9)

    print(f"\nFlow thresholds (from typical years {BOOTSTRAP_START_YEAR}-{BOOTSTRAP_END_YEAR}):")
    print(f"  95th percentile:  {p95:>10,.0f} ML/day")
    print(f"  99th percentile:  {p99:>10,.0f} ML/day")
    print(f"  99.9th percentile: {p999:>10,.0f} ML/day")

    # Count exceedances in historical typical years
    baseline_years = len(base_typical_only) / 365.25
    baseline_p95 = (base_typical_only['ML_day'] > p95).sum()
    baseline_p99 = (base_typical_only['ML_day'] > p99).sum()
    baseline_p999 = (base_typical_only['ML_day'] > p999).sum()

    print(f"\nHistorical typical years ({baseline_years:.1f} years):")
    print(f"  Days > 95th %ile:  {baseline_p95:>4} days  ({baseline_p95 / baseline_years:>5.1f} per year)")
    print(f"  Days > 99th %ile:  {baseline_p99:>4} days  ({baseline_p99 / baseline_years:>5.1f} per year)")
    print(f"  Days > 99.9th %ile: {baseline_p999:>4} days  ({baseline_p999 / baseline_years:>5.1f} per year)")

    # Count exceedances in synthetic data
    synth_years = len(synth_typical) / 365.25
    synth_p95 = (synth_typical['ML_day'] > p95).sum()
    synth_p99 = (synth_typical['ML_day'] > p99).sum()
    synth_p999 = (synth_typical['ML_day'] > p999).sum()

    print(f"\nSynthetic bootstrap ({synth_years:.1f} years):")
    print(f"  Days > 95th %ile:  {synth_p95:>4} days  ({synth_p95 / synth_years:>5.1f} per year)")
    print(f"  Days > 99th %ile:  {synth_p99:>4} days  ({synth_p99 / synth_years:>5.1f} per year)")
    print(f"  Days > 99.9th %ile: {synth_p999:>4} days  ({synth_p999 / synth_years:>5.1f} per year)")

    # Calculate frequency ratios
    ratio_p95 = (synth_p95 / synth_years) / (baseline_p95 / baseline_years) if baseline_p95 > 0 else 0
    ratio_p99 = (synth_p99 / synth_years) / (baseline_p99 / baseline_years) if baseline_p99 > 0 else 0
    ratio_p999 = (synth_p999 / synth_years) / (baseline_p999 / baseline_years) if baseline_p999 > 0 else 0

    print(f"\nFrequency ratio (synthetic / historical):")
    print(f"  95th %ile:  {ratio_p95:.2f}x")
    print(f"  99th %ile:  {ratio_p99:.2f}x")
    print(f"  99.9th %ile: {ratio_p999:.2f}x")

    print(f"\nValidation assessment:")
    if 0.8 <= ratio_p99 <= 1.2 and 0.7 <= ratio_p999 <= 1.3:
        print(f"  ✓ Extreme event frequencies are well-matched")
    elif ratio_p99 > 1.3 or ratio_p999 > 1.5:
        print(f"  ⚠ Bootstrap has TOO MANY extreme events")
        print(f"     Consider adjusting downweighting parameter (reduce from 0.3 to 0.2)")
    elif ratio_p99 < 0.7 or ratio_p999 < 0.5:
        print(f"  ⚠ Bootstrap has TOO FEW extreme events")
        print(f"     Consider increasing downweighting parameter (increase from 0.3 to 0.5)")
    else:
        print(f"  ~ Frequencies are acceptable")

    # Distribution comparison
    print(f"\nFlow distribution percentiles:")
    print(f"  Percentile    Historical    Synthetic    Ratio")
    print(f"  ----------    ----------    ----------   -----")
    for p in [50, 75, 90, 95, 99, 99.5, 99.9]:
        hist_val = np.percentile(base_typical_only['ML_day'], p)
        synth_val = np.percentile(synth_typical['ML_day'], p)
        ratio = synth_val / hist_val if hist_val > 0 else 0
        print(f"  {p:>6.1f}%     {hist_val:>10,.0f}    {synth_val:>10,.0f}   {ratio:>5.2f}x")

    # Visual verification by decade
    print(f"\n{'=' * 60}")
    print("EVENT COUNTS BY DECADE")
    print('=' * 60)

    synth_with_decade = synth_typical.copy()
    synth_with_decade['Decade'] = (synth_with_decade['Date'].dt.year // 10) * 10

    print(f"\nSynthetic data (events per decade):")
    for decade_start in range(2020, 2100, 10):
        decade_data = synth_with_decade[synth_with_decade['Decade'] == decade_start]
        if len(decade_data) > 0:
            years_in_decade = len(decade_data) / 365.25
            count_p99 = (decade_data['ML_day'] > p99).sum()
            count_p999 = (decade_data['ML_day'] > p999).sum()
            max_flow = decade_data['ML_day'].max()
            print(
                f"  {decade_start}s ({years_in_decade:.1f}y): {count_p99:3d} >99th %ile, {count_p999:2d} >99.9th %ile, max={max_flow:>10,.0f} ML/day")

    print(
        f"\nExpected per decade (~10 years): ~{baseline_p99 / baseline_years * 10:.0f} >99th %ile, ~{baseline_p999 / baseline_years * 10:.0f} >99.9th %ile")

    fdc_table = pd.DataFrame(fdc_data)
    fdc_csv_path = os.path.join(OUT_DIR, "fdc_all_scenarios.csv")
    fdc_table.to_csv(fdc_csv_path, index=False)
    print(f"Saved FDC table: {fdc_csv_path}")

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print('=' * 60)
    print("\nScenarios ready for reservoir modeling:")
    print(f"  • TYPICAL: {typical_path}")
    print(f"  • DROUGHT (10th): {drought_path}")
    print(f"  • SEVERE DROUGHT (5th): {severe_drought_path}")
    print(f"  • EXTREME RAIN (90th): {rain_path}")
    print(f"  • SEVERE RAIN (95th): {severe_rain_path}")
    print("\nUse insert_extreme_year.py to inject drought/rain into specific years.")


if __name__ == "__main__":
    main()