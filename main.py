# main.py — drought controls with calibrated presets (editable), Ravenswood-only check,
# dash for out-of-range DTE, pandas 'YE' fix, Streamlit state-safe widgets.
# Adds seepage + fluvial I/O, sidebar display of I/O params, and drought summary table.

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Config
BASE = Path(__file__).resolve().parent
JSON_FILE = BASE / "reservoir_system.json"
IGNORE = {".git", ".idea", "venv", "__pycache__", "outputs"}

DEFAULT_START = pd.Timestamp("2025-01-01")
DEFAULT_END   = pd.Timestamp("2029-12-31")
WARN_YEARS    = 15
CONFIRM_YEARS = 30
ALPHA = 4.5  # mm/day per kPa

# Tighten spacing a touch
st.markdown("""
<style>
div.element-container { margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# System + scenarios
@st.cache_resource(show_spinner=False)
def load_system(json_path: Path = JSON_FILE):
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)["system"]
    r1, r2 = cfg["reservoirs"]
    p01, p12 = cfg["pumps"]
    r1_name = r1.get("name", "Reservoir 1")
    r2_name = r2.get("name", "Reservoir 2")
    return r1, r2, p01, p12, r1_name, r2_name

def _reservoir_io_params(r: dict) -> tuple[float, float, float]:
    """Return (seepage_ML_day, fluvial_coeff_ML_per_mm, fluvial_min_mm) with safe defaults."""
    seep = float(r.get("losses", {}).get("seepage_ML_day", 0.0))
    fluv = r.get("inflows", {}).get("fluvial", {})
    coeff = float(fluv.get("coefficient_ML_per_mm", 0.0))
    rmin  = float(fluv.get("min_rain_mm", 0.0))
    return seep, coeff, rmin

@st.cache_resource(show_spinner=False)
def find_scenarios(base: str = ".") -> List[str]:
    out = []
    for name in sorted(os.listdir(base)):
        if name in IGNORE:
            continue
        pq = os.path.join(base, name, "raw_daily.parquet")
        if os.path.isfile(pq):
            out.append(name)
    return out[:6]

# --------------------------------------------------------------------
# Time/index handling
@st.cache_data(show_spinner=False)
def scenario_bounds(scn: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    pq = os.path.join(scn, "raw_daily.parquet")
    df = pd.read_parquet(pq, engine="pyarrow")
    if df.index.name == "time" and not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False)
        df = df.set_index("time")
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise KeyError(f"{scn}: expected time index or 'time' column")
    return df.index.min().normalize(), df.index.max().normalize()

@st.cache_data(show_spinner=False)
def bounds_for(scenarios: List[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    mins, maxs = [], []
    for scn in scenarios:
        gmin, gmax = scenario_bounds(scn)
        mins.append(gmin); maxs.append(gmax)
    return min(mins), max(maxs)

# --------------------------------------------------------------------
# Load drivers (strict Ravenswood verification)
@st.cache_data(show_spinner=False)
def load_pr_and_vpd(
    scn: str, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    pq = os.path.join(scn, "raw_daily.parquet")
    df = pd.read_parquet(pq, engine="pyarrow")

    req = {"tas_Ravenswood_degC","huss_Ravenswood_kgkg","psl_Ravenswood_Pa","pr_Ravenswood_mm_day"}
    missing = sorted(list(req - set(df.columns)))
    if missing:
        raise KeyError(f"{scn}: missing Ravenswood columns: {missing}")

    if df.index.name == "time" and not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False)
    elif "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=False)
        df = df.set_index("time")
    df = df.sort_index()

    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    if df.empty:
        raise ValueError(f"{scn}: no data in selected window")

    T = df["tas_Ravenswood_degC"].to_numpy(dtype=np.float32)
    q = df["huss_Ravenswood_kgkg"].to_numpy(dtype=np.float32)
    p = df["psl_Ravenswood_Pa"].to_numpy(dtype=np.float32)
    pr_mm = df["pr_Ravenswood_mm_day"].to_numpy(dtype=np.float32)

    e_pa = (q * p) / (0.622 + 0.378 * q)                 # Pa
    es_kpa = 0.6108 * np.exp((17.27 * T) / (T + 237.3))  # kPa
    vpd_kpa = np.maximum(0.0, es_kpa - (e_pa * 1e-3)).astype(np.float32)
    return df.index, pr_mm, vpd_kpa

# --------------------------------------------------------------------
# Calibration helpers
def _annual_stats(index: pd.DatetimeIndex, pr_mm: np.ndarray, vpd_kpa: np.ndarray) -> Tuple[pd.Series, pd.Series]:
    df = pd.DataFrame({"pr_mm": pr_mm, "vpd": vpd_kpa}, index=index)
    rain_y = df["pr_mm"].resample("YE").sum(min_count=300)
    vpd_y  = df["vpd"].resample("YE").mean()
    rain_y = rain_y.dropna()
    vpd_y  = vpd_y.loc[rain_y.index]
    return rain_y, vpd_y

def _return_period_multipliers(index: pd.DatetimeIndex,
                               pr_mm: np.ndarray,
                               vpd_kpa: np.ndarray,
                               T: int) -> Tuple[float, float]:
    rain_y, vpd_y = _annual_stats(index, pr_mm, vpd_kpa)
    if len(rain_y) < 10:
        return {50: (0.35, 1.15), 100: (0.30, 1.20), 200: (0.25, 1.25)}.get(T, (1.0, 1.0))
    p = 1.0 / float(T)
    q = 1.0 - p
    r_mean = float(rain_y.mean())
    v_mean = float(vpd_y.mean())
    r_q = float(np.quantile(rain_y, p))
    v_q = float(np.quantile(vpd_y, q))
    rain_mult = max(0.10, min(1.00, r_q / max(1e-6, r_mean)))
    evap_mult = max(1.00, min(1.50, v_q / max(1e-6, v_mean)))
    return rain_mult, evap_mult

def _driest_window_start(index: pd.DatetimeIndex, pr_mm: np.ndarray, window_days: int) -> pd.Timestamp:
    s = pd.Series(pr_mm, index=index)
    roll = s.rolling(window_days, min_periods=int(0.8*window_days)).sum()
    end = roll.idxmin()
    if pd.isna(end):
        return index[0]
    start = end - pd.Timedelta(days=window_days - 1)
    return max(index[0], start)

# --------------------------------------------------------------------
# Drought + DTE helpers
def apply_drought_window(index: pd.DatetimeIndex,
                         pr_mm: np.ndarray,
                         vpd_kpa: np.ndarray,
                         start: pd.Timestamp | None,
                         dur_days: int,
                         rain_mult: float,
                         evap_mult: float) -> tuple[np.ndarray, np.ndarray]:
    if start is None or dur_days <= 0:
        return pr_mm, vpd_kpa
    if abs(rain_mult - 1.0) < 1e-9 and abs(evap_mult - 1.0) < 1e-9:
        return pr_mm, vpd_kpa
    end = start + pd.Timedelta(days=int(dur_days) - 1)
    mask = (index >= start) & (index <= end)
    pr_s  = pr_mm.copy()
    vpd_s = vpd_kpa.copy()
    pr_s[mask]  = pr_s[mask]  * np.float32(rain_mult)
    vpd_s[mask] = vpd_s[mask] * np.float32(evap_mult)
    return pr_s, vpd_s

def add_days_to_empty(sim: pd.DataFrame, demand: float) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    met = sim["Demand_met_ML_day"].to_numpy(dtype=np.float32)
    shortage = met + 1e-6 < np.float32(demand)
    n = shortage.shape[0]
    next_fail = np.full(n, -1, dtype=np.int32)
    last = -1
    for i in range(n - 1, -1, -1):
        if shortage[i]:
            last = i
        next_fail[i] = last
    dte = np.full(n, np.inf, dtype=np.float32)
    for i in range(n):
        if next_fail[i] != -1:
            dte[i] = (next_fail[i] - i + 1)
    sim["Days_to_Empty"] = dte
    sim["Runout_Flag"] = shortage
    runout_date = sim.index[next_fail[0]] if next_fail[0] != -1 else None
    return sim, runout_date

# --------------------------------------------------------------------
# Simulator (now includes seepage + fluvial)
def simulate_from_pr_vpd(index: pd.DatetimeIndex,
                         pr_mm: np.ndarray, vpd_kpa: np.ndarray,
                         demand_ml_day: float,
                         A1_m2: float, A2_m2: float,
                         cap1: float, cap2: float,
                         s1_0: float, s2_0: float,
                         pump01_cap: float, pump12_cap: float,
                         seep1_ml_day: np.ndarray, seep2_ml_day: np.ndarray,
                         fluv1_ml_day: np.ndarray, fluv2_ml_day: np.ndarray,
                         s1_min: float = 0.0, s2_min: float = 0.0) -> pd.DataFrame:
    f32 = np.float32
    n = pr_mm.shape[0]
    demand = f32(demand_ml_day)
    cap1 = f32(cap1); cap2 = f32(cap2)
    s1 = f32(s1_0);  s2 = f32(s2_0)
    p01cap = f32(pump01_cap); p12cap = f32(pump12_cap)
    s1min = f32(s1_min); s2min = f32(s2_min)

    mm_to_ml1 = f32(A1_m2 / 1_000_000.0)
    mm_to_ml2 = f32(A2_m2 / 1_000_000.0)
    evap_mm = (f32(ALPHA) * vpd_kpa.astype(f32))
    rain1_ml = pr_mm.astype(f32) * mm_to_ml1
    rain2_ml = pr_mm.astype(f32) * mm_to_ml2
    evap1_ml = evap_mm * mm_to_ml1
    evap2_ml = evap_mm * mm_to_ml2

    # Net daily mass balance: rainfall + fluvial - evaporation - seepage
    m1 = rain1_ml + fluv1_ml_day.astype(f32) - evap1_ml - seep1_ml_day.astype(f32)
    m2 = rain2_ml + fluv2_ml_day.astype(f32) - evap2_ml - seep2_ml_day.astype(f32)

    S1 = np.empty(n, dtype=f32); S2 = np.empty(n, dtype=f32)
    P01 = np.zeros(n, dtype=f32); P12 = np.zeros(n, dtype=f32)
    Dmet = np.zeros(n, dtype=f32)

    for t in range(n):
        free2_pre = max(f32(0), cap2 - (s2 + m2[t]))
        P12_des = min(p12cap, free2_pre)
        spare1_if_P12 = max(f32(0), cap1 - (s1 + m1[t] - P12_des))
        P01_max = min(p01cap, spare1_if_P12)
        r1_supply_total = max(f32(0), s1 - s1min) + m1[t] + P01_max
        P12_t = min(P12_des, max(f32(0), r1_supply_total))
        spare1_actual = max(f32(0), cap1 - (s1 + m1[t] - P12_t))
        P01_t = min(p01cap, spare1_actual)
        availR2 = max(f32(0), s2 - s2min) + m2[t] + P12_t
        Dmet_t = min(demand, availR2)
        s1 = min(cap1, max(s1min, s1 + m1[t] + P01_t - P12_t))
        s2 = min(cap2, max(s2min, s2 + m2[t] + P12_t - Dmet_t))

        S1[t] = s1; S2[t] = s2
        P01[t] = P01_t; P12[t] = P12_t; Dmet[t] = Dmet_t

    return pd.DataFrame(
        {
            "S1_ML": S1, "S2_ML": S2,
            "P01_ML_day": P01, "P12_ML_day": P12,
            "Demand_met_ML_day": Dmet,
            "Rain_R1_ML_day": rain1_ml, "Evap_R1_ML_day": evap1_ml,
            "Rain_R2_ML_day": rain2_ml, "Evap_R2_ML_day": evap2_ml,
            "Seep_R1_ML_day": seep1_ml_day.astype(f32), "Seep_R2_ML_day": seep2_ml_day.astype(f32),
            "Fluvial_R1_ML_day": fluv1_ml_day.astype(f32), "Fluvial_R2_ML_day": fluv2_ml_day.astype(f32),
        },
        index=index,
    )

@st.cache_data(show_spinner=False)
def run_one_scenario(
    scn: str, start: pd.Timestamp | None, end: pd.Timestamp | None,
    demand: float, A1: float, A2: float, cap1: float, cap2: float,
    s1_0: float, s2_0: float, pump01: float, pump12: float,
    drought_mode: str,
    drought_start: pd.Timestamp | None, drought_days: int,
    rain_mult: float, evap_mult: float,
) -> pd.DataFrame:
    idx, pr_mm, vpd_kpa = load_pr_and_vpd(scn, start, end)

    applied_mode = "None"
    applied_rmult = 1.0
    applied_emult = 1.0

    if drought_mode != "None":
        pr_mm, vpd_kpa = apply_drought_window(
            idx, pr_mm, vpd_kpa,
            pd.to_datetime(drought_start) if drought_start is not None else None,
            int(drought_days), float(rain_mult), float(evap_mult)
        )
        applied_mode, applied_rmult, applied_emult = drought_mode, float(rain_mult), float(evap_mult)

    # Pull IO params from JSON
    r1_json, r2_json, _, _, _, _ = load_system()
    seep1, coeff1, min1 = _reservoir_io_params(r1_json)
    seep2, coeff2, min2 = _reservoir_io_params(r2_json)

    # Daily arrays
    n = pr_mm.shape[0]
    seep1_arr = np.full(n, np.float32(seep1))
    seep2_arr = np.full(n, np.float32(seep2))
    # Fluvial only when rain ≥ threshold.  No subtraction of threshold from depth.
    fluv1_arr = (pr_mm * np.float32(coeff1)) * (pr_mm >= np.float32(min1))
    fluv2_arr = (pr_mm * np.float32(coeff2)) * (pr_mm >= np.float32(min2))
    fluv1_arr = fluv1_arr.astype(np.float32)
    fluv2_arr = fluv2_arr.astype(np.float32)

    sim = simulate_from_pr_vpd(
        idx, pr_mm, vpd_kpa,
        demand, A1, A2, cap1, cap2, s1_0, s2_0, pump01, pump12,
        seep1_arr, seep2_arr, fluv1_arr, fluv2_arr
    )
    sim, runout_date = add_days_to_empty(sim, demand)
    sim.attrs["runout_date"] = runout_date
    sim.attrs["drought_mode"] = applied_mode
    sim.attrs["rain_mult"] = applied_rmult
    sim.attrs["evap_mult"] = applied_emult
    return sim

# --------------------------------------------------------------------
# UI helpers
def small_kv(label: str, value: str):
    st.markdown(
        f"<div style='font-size:0.9rem; line-height:1.2'><b>{label}:</b> {value}</div>",
        unsafe_allow_html=True,
    )

def thin_for_chart(df: pd.DataFrame, target_points: int = 1000) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    step = max(1, len(df) // target_points)
    return df.iloc[::step]

def build_overlays(results: Dict[str, pd.DataFrame], r1_name: str, r2_name: str) -> Dict[str, pd.DataFrame]:
    # Keep existing overlays for storage, rain, evap only.  No new seepage/fluvial overlays.
    storage_series, rn_cols, ev_cols = [], [], []
    for scn, df in results.items():
        if isinstance(df, Exception):
            continue
        storage_series.append(
            df[["S1_ML", "S2_ML"]].rename(
                columns={"S1_ML": f"{r1_name} — {scn}",
                         "S2_ML": f"{r2_name} — {scn}"}
            )
        )
        rn_cols += [df["Rain_R1_ML_day"].rename(f"Rain — {r1_name} — {scn}"),
                    df["Rain_R2_ML_day"].rename(f"Rain — {r2_name} — {scn}")]
        ev_cols += [df["Evap_R1_ML_day"].rename(f"Evap — {r1_name} — {scn}"),
                    df["Evap_R2_ML_day"].rename(f"Evap — {r2_name} — {scn}")]
    ovs = {
        "storage": pd.concat(storage_series, axis=1) if storage_series else pd.DataFrame(),
        "rain":    pd.concat(rn_cols, axis=1) if rn_cols else pd.DataFrame(),
        "evap":    pd.concat(ev_cols, axis=1) if ev_cols else pd.DataFrame(),
    }
    return {k: thin_for_chart(v) for k, v in ovs.items()}

def build_tables(results: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    keep_cols = ["S1_ML", "S2_ML", "P01_ML_day", "P12_ML_day",
                 "Demand_met_ML_day",
                 "Rain_R1_ML_day", "Evap_R1_ML_day", "Seep_R1_ML_day", "Fluvial_R1_ML_day",
                 "Rain_R2_ML_day", "Evap_R2_ML_day", "Seep_R2_ML_day", "Fluvial_R2_ML_day",
                 "Days_to_Empty", "Runout_Flag"]
    daily: Dict[str, pd.DataFrame] = {}
    weekly: Dict[str, pd.DataFrame] = {}
    for scn, df in results.items():
        if isinstance(df, Exception):
            continue
        d = df[keep_cols].replace(np.inf, np.nan)  # display blank for inf
        w = d.resample("W", label="left", closed="left").agg({
            "S1_ML": "last", "S2_ML": "last",
            "P01_ML_day": "sum", "P12_ML_day": "sum",
            "Demand_met_ML_day": "sum",
            "Rain_R1_ML_day": "sum", "Evap_R1_ML_day": "sum", "Seep_R1_ML_day": "sum", "Fluvial_R1_ML_day": "sum",
            "Rain_R2_ML_day": "sum", "Evap_R2_ML_day": "sum", "Seep_R2_ML_day": "sum", "Fluvial_R2_ML_day": "sum",
            "Days_to_Empty": "min",
            "Runout_Flag": "max",
        })
        daily[scn] = d
        weekly[scn] = w
    return {"Daily": daily, "Weekly": weekly}

def combine_tables(tables_for_resolution: Dict[str, pd.DataFrame],
                   selected: List[str]) -> pd.DataFrame:
    frames = []
    for scn in selected:
        tbl = tables_for_resolution.get(scn)
        if tbl is None or tbl.empty:
            continue
        frames.append(tbl.assign(Scenario=scn))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    cols = ["Scenario"] + [c for c in out.columns if c != "Scenario"]
    return out[cols]

# --------------------------------------------------------------------
# App
def run_app():
    st.set_page_config(page_title="Two-Reservoir Model", layout="wide")
    st.title("Reservoir Model with Climate Drivers")

    r1, r2, p01, p12, r1_name, r2_name = load_system()
    all_scens = find_scenarios(".")
    if not all_scens:
        st.error("No scenarios found with raw_daily.parquet")
        st.stop()

    # Pull IO params for sidebar display
    r1_seep, r1_fluv_coeff, r1_fluv_min = _reservoir_io_params(r1)
    r2_seep, r2_fluv_coeff, r2_fluv_min = _reservoir_io_params(r2)

    # Session storage
    for k, v in [("results", {}), ("overlays", {}), ("tables", {}), ("render_params", {}), ("last_preset", "None")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Placeholders
    storage_ph = st.empty(); rain_ph = st.empty(); evap_ph = st.empty(); table_ph = st.empty(); drought_tbl_ph = st.empty()
    if not st.session_state.results:
        storage_ph.markdown("<div style='height:380px'></div>", unsafe_allow_html=True)
        rain_ph.markdown("<div style='height:260px'></div>", unsafe_allow_html=True)
        evap_ph.markdown("<div style='height:260px'></div>", unsafe_allow_html=True)
        table_ph.markdown("<div style='height:400px'></div>", unsafe_allow_html=True)
        drought_tbl_ph.markdown("<div style='height:140px'></div>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.subheader("Reservoir properties")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div style='font-size:1.2rem; font-weight:600'>{r1_name}</div>", unsafe_allow_html=True)
            small_kv("Capacity (ML)", f"{float(r1['capacity_ML']):,.0f}")
            small_kv("Area (m²)", f"{float(r1['surface_area_m²']):,.0f}")
            small_kv("Seepage (ML/d)", f"{r1_seep:.3f}")
            small_kv("Fluvial coeff (ML/mm)", f"{r1_fluv_coeff:.3f}")
            small_kv("Fluvial min rain (mm)", f"{r1_fluv_min:.1f}")
        with c2:
            st.markdown(f"<div style='font-size:1.2rem; font-weight:600'>{r2_name}</div>", unsafe_allow_html=True)
            small_kv("Capacity (ML)", f"{float(r2['capacity_ML']):,.0f}")
            small_kv("Area (m²)", f"{float(r2['surface_area_m²']):,.0f}")
            small_kv("Seepage (ML/d)", f"{r2_seep:.3f}")
            small_kv("Fluvial coeff (ML/mm)", f"{r2_fluv_coeff:.3f}")
            small_kv("Fluvial min rain (mm)", f"{r2_fluv_min:.1f}")

        st.subheader("Scenarios")
        selected: List[str] = []
        for scn in all_scens:
            default_checked = (scn == "SSP1-26")
            if st.checkbox(scn, value=default_checked, key=f"scn_{scn}"):
                selected.append(scn)
        if not selected:
            st.warning("Tick at least one scenario.")
            st.stop()

        resolution = st.radio("Resolution", ["Weekly", "Daily"], index=0, horizontal=True)

        # Dates + params
        gmin_sel, gmax_sel = bounds_for(selected)
        date_min, date_max = gmin_sel.date(), gmax_sel.date()

        with st.form("params"):
            def_start = max(gmin_sel, DEFAULT_START).date()
            def_end   = min(gmax_sel, DEFAULT_END).date()
            if def_start > def_end:
                def_start, def_end = date_min, date_max

            start_input = st.date_input("Start date", value=def_start,
                                        min_value=date_min, max_value=date_max, format="YYYY-MM-DD", key="start_date_widget")
            end_input   = st.date_input("End date", value=def_end,
                                        min_value=date_min, max_value=date_max, format="YYYY-MM-DD", key="end_date_widget")
            start_date, end_date = map(pd.to_datetime, [start_input, end_input])
            if start_date > end_date:
                st.error("Start date must be on or before End date.")

            span_days = max(0, (end_date - start_date).days + 1)
            span_years = span_days / 365.25 if span_days > 0 else 0.0
            st.caption(f"Selected span ~ {span_years:.1f} years  ({span_days} days).")
            too_long = span_years > WARN_YEARS
            needs_confirm = span_years > CONFIRM_YEARS
            confirm_long = False
            if too_long and not needs_confirm:
                st.warning("Large time window selected.  This may be slow to render.")
            if needs_confirm:
                st.error("Very large time window selected.  Expect slow performance.")
                confirm_long = st.checkbox("I understand and want to run this long window", value=False)

            demand = st.number_input("Outflow demand (ML/day)",
                                     min_value=0.0, max_value=20.0, value=10.0, step=0.1, format="%.1f")

            st.divider()
            pump01 = st.number_input(f"River → {r1_name} cap (ML/day)",
                                     value=float(p01["max_rate_ML_day"]), min_value=0.0, step=0.1, format="%.1f")
            pump12 = st.number_input(f"{r1_name} → {r2_name} cap (ML/day)",
                                     value=float(p12["max_rate_ML_day"]), min_value=0.0, step=0.1, format="%.1f")

            st.divider()
            cap1 = float(r1["capacity_ML"]); cap2 = float(r2["capacity_ML"])
            s1_0 = st.number_input(f"{r1_name} initial (ML)",
                                   value=int(round(r1["initial_level_ML"])), min_value=0, max_value=int(round(cap1)),
                                   step=1, format="%d")
            s2_0 = st.number_input(f"{r2_name} initial (ML)",
                                   value=int(round(r2["initial_level_ML"])), min_value=0, max_value=int(round(cap2)),
                                   step=1, format="%d")
            A1 = float(r1["surface_area_m²"]); A2 = float(r2["surface_area_m²"])

            # Drought preset + editable controls
            st.divider()
            preset = st.selectbox("Drought", ["None", "1 in 50 years", "1 in 100 years", "1 in 200 years", "Custom"],
                                  index=["None","1 in 50 years","1 in 100 years","1 in 200 years","Custom"].index(st.session_state.get("last_preset","None")),
                                  key="drought_preset")

            # Ensure defaults exist in session_state before creating widgets that read them
            st.session_state.setdefault("drought_start", def_start)
            st.session_state.setdefault("drought_days", 365)
            st.session_state.setdefault("drought_rain", 1.0)
            st.session_state.setdefault("drought_evap", 1.0)

            # Refill controls on preset change
            if preset != st.session_state.last_preset:
                if preset in {"1 in 50 years","1 in 100 years","1 in 200 years"} and selected:
                    try:
                        idx0, pr0, vpd0 = load_pr_and_vpd(selected[0], start_date, end_date)
                        Tret = int(preset.split("/")[1])
                        rmult, emult = _return_period_multipliers(idx0, pr0, vpd0, Tret)
                        dur = 365
                        start_d = _driest_window_start(idx0, pr0, dur)
                        st.session_state["drought_days"]  = dur
                        st.session_state["drought_start"] = start_d.date()
                        st.session_state["drought_rain"]  = float(rmult)
                        st.session_state["drought_evap"]  = float(emult)
                    except Exception:
                        fallback = {"1 in 50 years": (0.35, 1.15), "1 in 100 years": (0.30, 1.20), "1 in 200 years": (0.25, 1.25)}
                        st.session_state["drought_days"]  = 365
                        st.session_state["drought_start"] = def_start
                        st.session_state["drought_rain"]  = fallback[preset][0]
                        st.session_state["drought_evap"]  = fallback[preset][1]
                elif preset == "Custom":
                    st.session_state["drought_start"] = def_start
                    st.session_state["drought_days"]  = 365
                    st.session_state["drought_rain"]  = 1.00
                    st.session_state["drought_evap"]  = 1.00
                else:  # None
                    st.session_state["drought_days"]  = 0
                    st.session_state["drought_rain"]  = 1.00
                    st.session_state["drought_evap"]  = 1.00
                st.session_state.last_preset = preset

            # Widgets — do not pass value= when using keys that are in session_state
            d_start = st.date_input(
                "Drought start",
                min_value=date_min, max_value=date_max, format="YYYY-MM-DD",
                key="drought_start",
            )
            d_days = st.number_input(
                "Drought duration (days)",
                min_value=0, max_value=3650, step=30,
                key="drought_days",
            )
            r_mult = st.slider(
                "Rainfall multiplier",
                0.0, 1.0, step=0.05,
                key="drought_rain",
            )
            e_mult = st.slider(
                "Evap multiplier",
                1.0, 2.0, step=0.05,
                key="drought_evap",
            )

            run_clicked = st.form_submit_button("Run")

    # Pending hint
    current_params = dict(
        selected=tuple(selected), resolution=resolution,
        start_date=str(start_date.date()), end_date=str(end_date.date()),
        demand=float(demand), pump01=float(pump01), pump12=float(pump12),
        cap1=float(cap1), cap2=float(cap2), s1_0=int(s1_0), s2_0=int(s2_0),
        A1=float(A1), A2=float(A2),
        drought_mode=preset,
        drought_start=str(pd.to_datetime(st.session_state.get("drought_start")).date()),
        drought_days=int(st.session_state.get("drought_days")),
        rain_mult=float(st.session_state.get("drought_rain")),
        evap_mult=float(st.session_state.get("drought_evap")),
    )
    if st.session_state.render_params and current_params != st.session_state.render_params:
        st.caption("Changes pending.  Press Run to apply.")

    if (run_clicked and start_date > end_date):
        run_clicked = False
        st.info("Run cancelled.  Fix the date range.")
    if run_clicked and needs_confirm and not confirm_long:
        run_clicked = False
        st.info("Run cancelled.  Reduce the window or tick the confirmation box.")

    # Compute
    if run_clicked:
        with st.spinner("Running…"):
            results: Dict[str, pd.DataFrame] = {}
            for scn in selected:
                try:
                    sim = run_one_scenario(
                        scn, start_date, end_date,
                        float(demand), A1, A2, cap1, cap2, float(s1_0), float(s2_0),
                        float(pump01), float(pump12),
                        drought_mode=preset,
                        drought_start=pd.to_datetime(st.session_state.get("drought_start")) if preset != "None" else None,
                        drought_days=int(st.session_state.get("drought_days")),
                        rain_mult=float(st.session_state.get("drought_rain")),
                        evap_mult=float(st.session_state.get("drought_evap")),
                    )
                    results[scn] = sim
                except Exception as e:
                    results[scn] = e

        st.session_state.results = results
        st.session_state.render_params = current_params
        st.session_state.overlays = build_overlays(results, r1_name, r2_name)
        st.session_state.tables = build_tables(results)
        st.success("Done.")

    # Nothing yet
    if not st.session_state.results:
        st.stop()

    # Charts
    ov = st.session_state.get("overlays", {})
    storage_df = ov.get("storage"); rain_df = ov.get("rain"); evap_df = ov.get("evap")

    if storage_df is not None and not storage_df.empty:
        storage_ph.empty()
        with storage_ph.container():
            st.subheader("Storage (ML)")
            st.line_chart(storage_df, height=360)

    if rain_df is not None and not rain_df.empty:
        rain_ph.empty()
        with rain_ph.container():
            with st.expander("Rainfall (ML/day)", expanded=False):
                st.line_chart(rain_df, height=240)

    if evap_df is not None and not evap_df.empty:
        evap_ph.empty()
        with evap_ph.container():
            with st.expander("Evaporation (ML/day)", expanded=False):
                st.line_chart(evap_df, height=240)

    # Drought summary table (replaces per-scenario text boxes)
    drought_tbl_ph.empty()
    with drought_tbl_ph.container():
        rows = []
        window_len_days = (pd.to_datetime(current_params["end_date"]) - pd.to_datetime(current_params["start_date"])).days + 1
        for scn in selected:
            df_res = st.session_state.results.get(scn)
            if isinstance(df_res, Exception) or df_res is None or df_res.empty:
                continue
            runout = df_res.attrs.get("runout_date")
            dte0 = float(df_res["Days_to_Empty"].iloc[0])
            dte_str = "" if np.isinf(dte0) or (dte0 > window_len_days) else f"{int(dte0):,d}"
            mode = df_res.attrs.get("drought_mode", "None")
            rmult = df_res.attrs.get("rain_mult", 1.0)
            emult = df_res.attrs.get("evap_mult", 1.0)
            rows.append({
                "Scenario": scn,
                "Drought mode": mode,
                "Rain ×": f"{rmult:.2f}",
                "Evap ×": f"{emult:.2f}",
                "Days to Empty (t₀)": dte_str if dte_str else "—",
                "First shortfall": "None" if runout is None else str(runout.date()),
            })
        if rows:
            st.subheader("Drought summary")
            st.dataframe(pd.DataFrame(rows), height=160)
        else:
            st.info("No drought summary to show.")

    # Table
    tables_by_res = st.session_state.tables.get(resolution, {})
    combined = combine_tables(tables_by_res, selected)
    table_ph.empty()
    with table_ph.container():
        st.subheader(f"Summary — {resolution}")
        if combined.empty:
            st.info("No data to show.")
        else:
            max_rows = 500 if resolution == "Daily" else 300
            st.dataframe(combined.round(3).tail(max_rows), width="stretch", height=380)

# --------------------------------------------------------------------
if __name__ == "__main__":
    run_app()
