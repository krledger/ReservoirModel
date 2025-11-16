"""
Tab 2: Actual Historical Data
Handles display of actual observed reservoir data
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from reservoir_operations import calculate_statistics
from reservoir_plotting import create_plots


def render_historical_tab(system, reservoirs, pumps, historical_results, hist_min_year, hist_max_year):
    """Render the historical data tab"""
    
    st.header("Actual Historical Data")

    # Get date range from sidebar
    start_year_hist = st.session_state.get('hist_start_year', max(hist_min_year, 2020))
    end_year_hist = st.session_state.get('hist_end_year', hist_max_year)

    if historical_results is not None:
        try:
            results = historical_results.copy()

            # Filter to date range
            start_date = f"{start_year_hist}-01-01"
            end_date = f"{end_year_hist}-12-31"
            mask = (results.index >= start_date) & (results.index <= end_date)
            results = results.loc[mask]

            if len(results) == 0:
                st.warning("No data available for selected date range")
            else:
                # Load river flow data
                results = load_wmip_river_flow(results)
                
                # Load AGCD climate data
                results, agcd_loaded = load_agcd_climate_data(results, reservoirs)
                
                # Prepare results for plotting
                results = prepare_historical_results(results)
                
                # Create scenario description
                scenario_desc = "HISTORICAL DATA"
                if agcd_loaded:
                    scenario_desc += " + AGCD"
                
                # Calculate statistics
                stats = calculate_statistics(results, reservoirs, pumps)
                

                # Display statistics
                display_historical_statistics(stats, len(results))

                # Create plots
                r1_turkeys = system.system_config['system']['reservoirs'][0]
                r2_surhs = system.system_config['system']['reservoirs'][1]
                r1_min = reservoirs[0].get('min_capacity_ML', 0)
                r2_min = reservoirs[1].get('min_capacity_ML', 0)
                default_river_pump_low = pumps[0]['cutoffs']['low_flow_ML_day']
                default_river_pump_high = pumps[0]['cutoffs']['high_flow_ML_day']

                fig = create_plots(results, scenario_desc, start_year_hist, end_year_hist, r1_min, r2_min,
                                   default_river_pump_low, default_river_pump_high, r1_turkeys, r2_surhs)
                st.plotly_chart(fig, use_container_width=True)

                # Download results
                with st.expander("💾 Download Results"):
                    csv = results.to_csv()
                    st.download_button(
                        label="Download historical data (CSV)",
                        data=csv,
                        file_name=f"historical_data_{start_year_hist}_{end_year_hist}.csv",
                        mime="text/csv",
                        key="download_hist"
                    )

        except Exception as e:
            st.error(f"Error displaying historical data: {e}")
            st.exception(e)
    else:
        st.warning("No historical data available. Please load historical_data_consolidated.csv")


def load_wmip_river_flow(results):
    """Load river flow data from WMIP and join to results"""
    with st.spinner('Loading river flow data from WMIP...'):
        try:
            flow_base = Path('.') / 'wmipData'
            typical_file = flow_base / 'flows_bootstrap_typical.parquet'

            if typical_file.exists():
                flow_data = pd.read_parquet(typical_file)

                if 'Date' in flow_data.columns:
                    flow_data['Date'] = pd.to_datetime(flow_data['Date'])
                    flow_data = flow_data.set_index('Date')
                elif not isinstance(flow_data.index, pd.DatetimeIndex):
                    flow_data.index = pd.to_datetime(flow_data.index)

                # Filter flow data to match results date range
                flow_mask = (flow_data.index >= results.index.min()) & (flow_data.index <= results.index.max())
                flow_data_filtered = flow_data.loc[flow_mask]

                # Extract river flow column
                if 'ML_day' in flow_data_filtered.columns:
                    river_flow_col = 'ML_day'
                elif 'CUMECS' in flow_data_filtered.columns:
                    flow_data_filtered['ML_day'] = flow_data_filtered['CUMECS'] * 86.4
                    river_flow_col = 'ML_day'
                else:
                    river_flow_col = None

                # Join river flow to results
                if river_flow_col:
                    results = results.join(flow_data_filtered[[river_flow_col]], how='left', rsuffix='_wmip')
                    if 'river_flow_ML_day' not in results.columns or results['river_flow_ML_day'].isna().all():
                        results['river_flow_ML_day'] = results[river_flow_col]
            else:
                pass  # No WMIP data found
        except Exception as e:
            pass  # Error loading WMIP data
    return results


def load_agcd_climate_data(results, reservoirs):
    """Load AGCD climate data and calculate derived variables"""
    agcd_loaded = False
    
    with st.spinner('Loading AGCD climate data...'):
        try:
            agcd_dir = Path('.') / 'metricsDataFiles' / 'AGCD'
            agcd_file = agcd_dir / 'raw_daily.parquet'

            if agcd_file.exists():
                agcd_data = pd.read_parquet(agcd_file)
                agcd_data.index = pd.to_datetime(agcd_data.index)

                # Filter to match results date range
                agcd_mask = (agcd_data.index >= results.index.min()) & (agcd_data.index <= results.index.max())
                agcd_filtered = agcd_data.loc[agcd_mask]

                # Map AGCD column names
                climate_data = {}

                # Temperature
                temp_col = find_climate_column(agcd_filtered.columns, 'tas', 'degC', 'Ravenswood')
                if temp_col:
                    climate_data['temperature_degC'] = agcd_filtered[temp_col]

                # Humidity
                huss_col = find_climate_column(agcd_filtered.columns, 'huss', 'g_per_kg', 'Ravenswood')
                if huss_col:
                    climate_data['specific_humidity_g_kg'] = agcd_filtered[huss_col]

                # Precipitation
                pr_col = find_climate_column(agcd_filtered.columns, 'pr', 'mm', 'Ravenswood')
                if pr_col:
                    climate_data['precipitation_mm_day'] = agcd_filtered[pr_col]

                # Wind speed
                wind_col = find_climate_column(agcd_filtered.columns, 'wind', None, 'Ravenswood')
                if wind_col:
                    climate_data['wind_speed_ms'] = agcd_filtered[wind_col]

                if climate_data:
                    # Create dataframe and join
                    climate_df = pd.DataFrame(climate_data)
                    results = results.join(climate_df, how='left')

                    # Calculate derived variables
                    if 'temperature_degC' in results.columns and 'specific_humidity_g_kg' in results.columns:
                        results = calculate_climate_derived_variables(results, reservoirs)

                    agcd_loaded = True
                else:
                    pass  # No climate columns found
            else:
                pass  # AGCD file not found

        except Exception as e:
            pass  # Error loading AGCD data
    return results, agcd_loaded


def find_climate_column(columns, var_name, unit, location):
    """Find climate variable column prioritising specific location"""
    # Try exact match with location first
    if location:
        for col in columns:
            if location in col and var_name in col.lower():
                if unit is None or unit in col:
                    return col
    
    # Try without location
    for col in columns:
        if var_name in col.lower():
            if unit is None or unit in col:
                return col
    
    return None


def get_surface_area_safe(reservoir_config):
    """Safely get surface area from config with different possible key names"""
    # First, try to find any key that contains 'surface_area'
    for key in reservoir_config.keys():
        if 'surface_area' in key.lower():
            return reservoir_config[key]
    
    # Fallback to checking specific variations
    possible_keys = [
        'surface_area_m2',
        'surface_area_m²',
        'surface_area',
        'area_m2',
        'area'
    ]
    
    for key in possible_keys:
        if key in reservoir_config:
            return reservoir_config[key]
    
    # If no key found, raise informative error
    available_keys = list(reservoir_config.keys())
    raise KeyError(f"Could not find surface area key. Available keys: {available_keys}")


def calculate_climate_derived_variables(results, reservoirs):
    """Calculate relative humidity, evaporation and reservoir losses"""
    temp = results['temperature_degC'].values
    huss = results['specific_humidity_g_kg'].values

    # Only calculate where we have valid data
    valid_mask = ~(np.isnan(temp) | np.isnan(huss))
    if valid_mask.any():
        # Relative humidity
        es = 6.108 * np.exp((17.27 * temp) / (temp + 237.3))
        p = 1013.25
        e = (huss / 1000) * p / (0.622 + 0.378 * (huss / 1000))
        rh = (e / es) * 100
        results['relative_humidity_pct'] = np.clip(rh, 0, 100)

        # Evaporation
        es_evap = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        ea = (huss / 1000) * 101.325 / 0.622
        vpd = es_evap - ea
        vpd = np.maximum(vpd, 0)
        wind_speed = results.get('wind_speed_ms', pd.Series([2.0] * len(temp))).fillna(2.0).values
        evaporation_mm = 0.5 * (temp / 20) * vpd * (1 + 0.5 * wind_speed / 2)
        results['evaporation_mm_day'] = np.maximum(evaporation_mm, 0)

        # Reservoir evaporation losses - use safe getter
        r1_area = get_surface_area_safe(reservoirs[0])
        r2_area = get_surface_area_safe(reservoirs[1])
        results['r1_evap_ML_day'] = (results['evaporation_mm_day'] / 1000) * r1_area / 1000
        results['r2_evap_ML_day'] = (results['evaporation_mm_day'] / 1000) * r2_area / 1000
    
    return results


def prepare_historical_results(results):
    """Add columns expected by plotting code if missing"""
    for col in ['demand_deficit_ML', 'fluvial_inflow_ML', 'pluvial_inflow_r1_ML',
                'pluvial_inflow_r2_ML', 'total_inflow_ML', 'r1_evap_ML_day',
                'r2_evap_ML_day', 'temperature_degC', 'relative_humidity_pct',
                'precipitation_mm_day', 'river_flow_ML_day']:
        if col not in results.columns:
            if 'ML' in col or 'day' in col:
                results[col] = 0
            else:
                results[col] = np.nan
    
    return results


def display_historical_statistics(stats, num_days):
    """Display statistics for historical data"""
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Data Points", f"{num_days}")
    col2.metric("Avg Turkeys Nest Dam", f"{stats['avg_r1']:.1f} ML", 
                f"{int(stats['days_of_storage_r1'])} days")
    col3.metric("Avg Surhs Creek Dam", f"{stats['avg_r2']:.1f} ML", 
                f"{int(stats['days_of_storage_r2'])} days")
    col4.metric("Total Days of Storage", f"{int(stats['days_of_storage_total'])} days",
                help="Combined system storage at avg levels (demand + all losses, no inflows)")
    col5.metric("Min Surhs Creek Dam", f"{stats['min_r2']:.1f} ML")

    # Pump diagnostics
    st.markdown("### 🔧 Pump Activity Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Days TND→SCD=0", f"{stats['days_pump_r1_r2_zero']}",
                f"{stats['days_pump_r1_r2_zero'] / num_days * 100:.1f}% of time")
    col2.metric("Days River→TND Active", f"{stats['days_pump_river_r1_active']}",
                f"{stats['days_pump_river_r1_active'] / num_days * 100:.1f}% of time")
    col3.metric("Avg River→TND", f"{stats['avg_pump_river_r1']:.2f} ML/day")
    col4.metric("Avg TND→SCD", f"{stats['avg_pump_r1_r2']:.2f} ML/day")
    col5.metric("Avg SCD→Site", f"{stats['avg_pump_r2_site']:.2f} ML/day")

    # Pump utilization (annual)
    if 'utilization_river_r1_pct' in stats:
        st.markdown("#### Annual Utilisation")
        col1, col2, col3 = st.columns(3)
        col1.metric("River→TND", 
                   f"{stats['utilization_river_r1_pct']:.1f}%",
                   f"{stats['annual_vol_river_r1_ML']:.0f} ML/year")
        col2.metric("TND→SCD", 
                   f"{stats['utilization_r1_r2_pct']:.1f}%",
                   f"{stats['annual_vol_r1_r2_ML']:.0f} ML/year")
        col3.metric("SCD→Site", 
                   f"{stats['utilization_r2_site_pct']:.1f}%",
                   f"{stats['annual_vol_r2_site_ML']:.0f} ML/year")


def show_historical_data_info():
    """Show information about historical data requirements"""
    with st.expander("ℹ️ About Historical Data"):
        st.markdown("""
        ### Historical Data Source

        This tab displays actual observed data from the reservoir system.

        **To use this feature:**
        1. Run `consolidate_historical_data.py` to process your Excel file
        2. This creates `historical_data_consolidated.csv`
        3. Data will load automatically when the app starts

        **Data automatically loaded:**
        - ✓ **Reservoir levels** (Turkeys Nest Dam, Surhs Creek Dam) from CSV
        - ✓ **Pump flows** (River→TND, TND→SCD, SCD→Site) from CSV
        - ✓ **River flow** from WMIP data (if available)
        - ✓ **Climate data** from AGCD scenario (if available)

        **AGCD Climate Data:**
        - AGCD = Australian Gridded Climate Data (actual observations)
        - Same format as SSP scenarios: `metricsDataFiles/AGCD/raw_daily.parquet`
        - Downloaded/processed using same tools as SSP scenarios
        - Provides actual temperature, humidity, precipitation

        **Note:** Tab 2 shows only actual historical data.  For model predictions, use Tab 1.
        """)


def render_historical_sidebar(hist_min_year, hist_max_year):
    """Render sidebar controls for historical tab"""
    st.sidebar.header("Historical Data Controls")

    with st.sidebar.expander("📅 Date Range", expanded=True):
        start_year_hist = st.number_input("Start Year", min_value=hist_min_year, max_value=hist_max_year,
                                          value=max(hist_min_year, 2020), key="hist_start_year")
        end_year_hist = st.number_input("End Year", min_value=hist_min_year, max_value=hist_max_year,
                                        value=hist_max_year, key="hist_end_year")
