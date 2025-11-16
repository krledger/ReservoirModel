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
from reservoir_historical_tab import render_historical_tab, render_historical_sidebar
from reservoir_weekly_tab import render_weekly_tab, render_weekly_sidebar


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
def load_historical_data(hist_data_path):
    """Load historical data from CSV"""
    hist_file = Path(hist_data_path)
    if not hist_file.exists():
        return None, f"File not found: {hist_data_path}"

    try:
        results = pd.read_csv(hist_data_path, parse_dates=['date'], index_col='date')
        return results, None
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

        # Determine available year range from climate data
        min_year = system.climate_data.index.min().year
        max_year = system.climate_data.index.max().year

    except Exception as e:
                st.stop()

    # Initialise session state
    if 'initial_run_complete' not in st.session_state:
        st.session_state.initial_run_complete = False
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None

    # Load historical data at startup
    hist_data_path = "historical_data_consolidated.csv"
    historical_results, hist_error = load_historical_data(hist_data_path)

    # Extract presets from historical data
    presets = None
    if historical_results is not None:
        hist_min_year = historical_results.index.min().year
        hist_max_year = historical_results.index.max().year

        # Show data quality info
        with st.sidebar.expander("📊 Historical Data Quality", expanded=False):
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
    tab1, tab2, tab3 = st.tabs(["📊 Model Simulation", "📈 Actual Historical Data", "📋 Weekly Water Balance"])

    # Determine which tab is active
    active_tab = None

    # Tab 1: Model Simulation
    with tab1:
        active_tab = "model"
        render_model_sidebar(pumps, reservoirs, min_year, max_year, presets)
        render_model_tab(system, reservoirs, pumps, min_year, max_year, presets, historical_results)

    # Tab 2: Historical Data
    with tab2:
        if active_tab != "model":
            active_tab = "historical"
        render_historical_sidebar(hist_min_year, hist_max_year)
        render_historical_tab(system, reservoirs, pumps, historical_results, hist_min_year, hist_max_year)

    # Tab 3: Weekly Balance
    with tab3:
        if active_tab not in ["model", "historical"]:
            active_tab = "weekly"
        render_weekly_sidebar(historical_results)
        render_weekly_tab(historical_results, reservoirs, pumps)

    # System info at bottom
    render_system_info(reservoirs, pumps)


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
          - Select climate scenario: SSP1-26, SSP2-45, SSP5-85, or AGCD
          - Enable random variations for realistic demand/pump fluctuations
        - **Tab 2 (Actual):** View historical observations from your Excel data
          - Automatically loads AGCD climate data (actual observations)
          - Automatically loads WMIP river flow data
          - Shows complete actual system behaviour
        - **Tab 3 (Weekly Balance):** Weekly water balance reports
          - Select week ending date from dropdown
          - View current vs previous week comparisons
          - See days until storage depleted
        - **Compare:** Run Tab 1 with AGCD to compare model vs Tab 2 actual data
        - **Test scenarios:** Try drought/rain years to stress-test the system
        - **Modular design:** Code separated into logical modules for maintainability
        """)


if __name__ == "__main__":
    main()
