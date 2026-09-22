"""
Reservoir System Simulation Dashboard
Main application entry point
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Import modules
from reservoir_operations import ReservoirSystem, extract_presets_from_historical
from reservoir_model_tab import render_model_tab, render_model_sidebar
from reservoir_historical_tab import render_historical_tab
from reservoir_weekly_tab import render_weekly_tab, render_weekly_sidebar
from SiteActuals import SWB_PATH, actuals_weekly
from OpportunityTab import render_opportunity_tab
from CurrentReadingsTab import render_current_readings_tab
import CurrentReadings as cr
import ModelInputs as mi


# Set page config
st.set_page_config(page_title="Reservoir System Simulation", layout="wide")


@st.cache_resource
def load_system():
    """Load system config and climate data once"""
    system = ReservoirSystem(base_path='.')
    system.load_system_config()
    system.load_climate_data()
    return system


@st.cache_data
def load_site_actuals(swb_mtime):
    """Weekly site actuals from Weekly SWB.xlsx (read only).  The mtime argument refreshes the cache."""
    try:
        return actuals_weekly(), None
    except Exception as e:
        return None, str(e)


def main():
    """Main application logic"""
    
    # Load system
    try:
        with st.spinner('Loading system configuration...'):
            system = load_system()
        reservoirs = system.system_config['system']['reservoirs']
        pumps = system.system_config['system']['pumps']

        # Year range across all climate sources (AGCD from 1971, SSPs to 2100)
        min_year = mi.earliest_year()
        max_year = mi.latest_year()

    except Exception as e:
                st.stop()

    # Initialise session state
    if 'initial_run_complete' not in st.session_state:
        st.session_state.initial_run_complete = False
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None

    # Site actuals from the weekly site water balance workbook, with later entered readings appended
    if SWB_PATH.exists():
        workbook_actuals, hist_error = load_site_actuals(SWB_PATH.stat().st_mtime)
    else:
        workbook_actuals, hist_error = None, f"{SWB_PATH.name} not found"
    if hist_error:
        st.sidebar.warning(f"Site actuals not loaded: {hist_error}")
    historical_results = cr.merge_with_actuals(workbook_actuals)

    # Extract presets from historical data
    presets = None
    if historical_results is not None:
        hist_min_year = historical_results.index.min().year
        hist_max_year = historical_results.index.max().year

        # Show data quality info
        with st.sidebar.expander("📊 Site Actuals (Weekly SWB)", expanded=False):
            st.write(f"**Date Range:** {historical_results.index.min().strftime('%Y-%m-%d')} to {historical_results.index.max().strftime('%Y-%m-%d')}")
            st.write(f"**Total Records:** {len(historical_results)}")

            # Check for key columns
            key_cols = ['r1_level_ML', 'r2_level_ML', 'pump_river_to_r1_ML_day', 'pump_r1_to_r2_ML_day', 'pump_r2_to_site_ML_day']
            for col in key_cols:
                if col in historical_results.columns:
                    valid_count = historical_results[col].notna().sum()
                    pct = (valid_count / len(historical_results)) * 100
                    if pct < 50:
                        st.write(f"**{col}:** {valid_count} valid ({pct:.1f}%) ⚠️")
                    else:
                        st.write(f"**{col}:** {valid_count} valid ({pct:.1f}%)")

        try:
            presets = extract_presets_from_historical(historical_results)

            if presets:
                pass  # Presets loaded successfully
            else:
                pass  # Using default configuration values
        except Exception as e:
            presets = None
    else:
        hist_min_year = 2015
        hist_max_year = 2025

    # Title
    st.title("🌊 Reservoir System Simulation Dashboard")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Model Simulation", "📈 Actuals",
                                            "📋 Weekly Water Balance", "🎯 Actuals vs Potential",
                                            "📝 Current Readings"])

    # Determine which tab is active
    active_tab = None

    # One date range for the model and historical tabs
    render_date_range(min_year, max_year, presets)

    # Tab 1: Model Simulation
    with tab1:
        active_tab = "model"
        render_model_sidebar(pumps, reservoirs, min_year, max_year, presets, historical_results)
        render_model_tab(system, reservoirs, pumps, min_year, max_year, presets, historical_results)

    # Tab 2: Historical Data
    with tab2:
        if active_tab != "model":
            active_tab = "historical"
        render_historical_tab(system, reservoirs, pumps, historical_results, hist_min_year, hist_max_year)

    # Tab 3: Weekly Balance
    with tab3:
        if active_tab not in ["model", "historical"]:
            active_tab = "weekly"
        render_weekly_sidebar(historical_results)
        render_weekly_tab(historical_results, reservoirs, pumps)

    # Tab 4: Site actuals against as-operated and potential model runs
    with tab4:
        render_opportunity_tab()

    # Tab 5: Current readings and the printable water supply report
    with tab5:
        render_current_readings_tab(reservoirs, workbook_actuals, historical_results)

    # System info at bottom
    render_system_info(reservoirs, pumps)


def render_date_range(min_year, max_year, presets):
    """Single date range, shared by the model and historical tabs"""
    default_start = presets['model_start_year'] if presets else 2015
    default_end = presets['model_end_year'] if presets else 2025
    default_start = min(max(default_start, min_year), max_year)
    default_end = min(max(default_end, default_start), max_year)
    with st.sidebar.expander("📅 Date Range", expanded=True):
        st.number_input("Start Year", min_value=min_year, max_value=max_year,
                        value=st.session_state.get('start_year', default_start), key="start_year")
        st.number_input("End Year", min_value=min_year, max_value=max_year,
                        value=st.session_state.get('end_year', default_end), key="end_year")
        st.caption(f"Model inputs cover {min_year} to {max_year}.  The historical tab shows the part of "
                   f"this range its data covers.")


def render_system_info(reservoirs, pumps):
    """Render system architecture and tips"""

    with st.expander("⚙️ System Architecture"):
        st.markdown(f"""
        ### Reservoir System Flow

        ```
        RIVER (Burdekin)
           ↓ 
        [Pump: River→TND] (Max: {pumps[0]['max_rate_in_ML_day']} ML/day, operates when {pumps[0]['cutoffs']['low_flow_ML_day']} < flow < {pumps[0]['cutoffs']['high_flow_ML_day']} ML/day)
           ↓
        Turkeys Nest Dam (TND) - Capacity: {reservoirs[0]['capacity_ML']} ML, Min: {reservoirs[0].get('min_capacity_ML', 0)} ML
           ↓
        [Pump: TND→SCD] (Max: {pumps[0]['max_rate_out_ML_day']} ML/day)
           ↓
        Surhs Creek Dam (SCD) - Capacity: {reservoirs[1]['capacity_ML']} ML, Min: {reservoirs[1].get('min_capacity_ML', 0)} ML
           ↓  (+ Fluvial/Pluvial inflows)
        [Pump: SCD→Site] (Max: {pumps[1]['max_rate_out_ML_day']} ML/day)
           ↓
        DEMAND SITE (9.8 ML/day baseline)
        ```

        **Key Constraints:**
        - River pump only operates within specific flow window
        - TND must stay above {reservoirs[0].get('min_capacity_ML', 0)} ML (dead storage)
        - SCD must stay above {reservoirs[1].get('min_capacity_ML', 0)} ML (dead storage)
        - Final delivery pump limited to {pumps[1]['max_rate_out_ML_day']} ML/day
        """)

    with st.expander("💡 Tips"):
        st.markdown("""
        - **Auto-load on startup:** Tab 1 automatically loads presets from last 12 months of historical data
        - **Seamless transition:** Model starts where historical data ends, with actual reservoir levels
        - **Tab 1 (Model):** Run predictive simulations with different scenarios
          - Starts with presets from historical data (reservoir levels, demand)
          - Select the climate scenario after the AGCD record ends: SSP1-26, SSP2-45, SSP3-70 or SSP5-85
          - Enable random variations for realistic demand/pump fluctuations
        - **Tab 2 (Actuals):** Site actuals from Weekly SWB.xlsx and the entered readings
          - Automatically loads AGCD climate data (actual observations)
          - Automatically loads WMIP river flow data
          - Shows complete actual system behaviour
        - **Tab 3 (Weekly Balance):** Weekly water balance reports
          - Select week ending date from dropdown
          - View current vs previous week comparisons
          - See days until storage depleted
        - **Compare:** Run Tab 1 with AGCD to compare model vs Tab 2 actual data
        - **Tab 5 (Current Readings):** Enter the site weekly update and download the printable report
        - **Modular design:** Code separated into logical modules for maintainability
        """)


if __name__ == "__main__":
    main()
