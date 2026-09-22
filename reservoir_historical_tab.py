"""
Tab 2: Actuals
Site actuals from Weekly SWB.xlsx and the entered readings, from the first
reading onwards.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from reservoir_operations import calculate_statistics
import ReservoirPhysics as rp
from reservoir_plotting import create_plots, split_subplots, PLOTLY_CONFIG


def render_historical_tab(system, reservoirs, pumps, historical_results, hist_min_year, hist_max_year):
    """Render the historical data tab"""
    
    st.header("Actuals")
    if historical_results is None:
        st.warning("No site actuals available. Place Weekly SWB.xlsx in the project folder.")
        return

    # Shared date range from the sidebar, starting no earlier than the first actual reading
    levels = historical_results[['r1_level_ML', 'r2_level_ML']].dropna()
    first, last = levels.index.min(), levels.index.max()
    start_date = max(pd.Timestamp(f"{st.session_state.get('start_year', first.year)}-01-01"), first)
    end_date = min(pd.Timestamp(f"{st.session_state.get('end_year', last.year)}-12-31"), last)
    if start_date > end_date:
        st.info(f"Actuals run from {first:%d/%m/%Y} to {last:%d/%m/%Y}; the selected years are outside that.")
        return
    start_year_hist, end_year_hist = start_date.year, end_date.year
    st.caption(f"Showing {start_date:%d/%m/%Y} to {end_date:%d/%m/%Y}.  Dam volumes from {first:%d/%m/%Y}; pump "
               f"meters from August 2024.  Weeks after the workbook come from the entered readings.")

    if historical_results is not None:
        try:
            results = historical_results.copy()

            # Filter to date range
            mask = (results.index >= start_date) & (results.index <= end_date)
            results = results.loc[mask]

            if len(results) == 0:
                st.warning("No data available for selected date range")
            else:
                # Load river flow data
                results = load_wmip_river_flow(results)
                
                # Load AGCD climate data, then use the site rain gauge where the workbook has it
                results, agcd_loaded = load_agcd_climate_data(results, reservoirs)
                if 'site_rain_mm_day' in results.columns:
                    base = results['precipitation_mm_day'] if 'precipitation_mm_day' in results.columns else None
                    results['precipitation_mm_day'] = (results['site_rain_mm_day'] if base is None
                                                       else results['site_rain_mm_day'].combine_first(base))
                
                # Prepare results for plotting
                results = prepare_historical_results(results)
                
                # Create scenario description
                scenario_desc = "SITE ACTUALS (WEEKLY SWB)"
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
                st.markdown(f"**Reservoir System Analysis - {scenario_desc} ({start_year_hist}-{end_year_hist})**")
                for i, f in enumerate(split_subplots(fig)):
                    st.plotly_chart(f, config=PLOTLY_CONFIG, key=f"hist_chart_{i}")

                # Download results
                with st.expander("💾 Download Results"):
                    csv = results.to_csv()
                    st.download_button(
                        label="Download site actuals (CSV)",
                        data=csv,
                        file_name=f"site_actuals_{start_year_hist}_{end_year_hist}.csv",
                        mime="text/csv",
                        key="download_hist"
                    )

        except Exception as e:
            st.error(f"Error displaying historical data: {e}")
            st.exception(e)
    else:
        st.warning("No site actuals available. Place Weekly SWB.xlsx in the project folder.")


def load_wmip_river_flow(results):
    """Load river flow data from WMIP and join to results"""
    with st.spinner('Loading river flow data from WMIP...'):
        try:
            from ModelInputs import load_flows, FLOW_ACTUAL, SELLHEIM_FILE

            if SELLHEIM_FILE.exists():
                # Actual Sellheim record (the bootstrap file is synthetic after May 2024)
                flow_data = load_flows(FLOW_ACTUAL)

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
    """Join observed climate (AGCD, with humidity and wind filled by ModelInputs) to the results"""
    agcd_loaded = False
    try:
        from ModelInputs import load_climate, last_actual_climate_day
        c = load_climate()
        c = c[(c.index >= results.index.min()) & (c.index <= min(results.index.max(), last_actual_climate_day()))]
        if len(c):
            climate_df = pd.DataFrame({
                'temperature_degC': c['tas_Ravenswood_degC'],
                'temperature_range_degC': c['dtr_Ravenswood_degC'],
                'specific_humidity_g_kg': c['huss_Ravenswood_g_per_kg'],
                'precipitation_mm_day': c['pr_Ravenswood_mm_day'],
                'wind_speed_ms': c['wind_Ravenswood_ms'],
            })
            results = results.join(climate_df, how='left')
            results = calculate_climate_derived_variables(results, reservoirs)
            agcd_loaded = True
    except Exception as e:
        st.caption(f"AGCD climate not loaded: {e}")
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

        # FAO-56 open water evaporation
        wind_speed = results['wind_speed_ms'].fillna(2.0).values if 'wind_speed_ms' in results else None
        dtr = results['temperature_range_degC'].values if 'temperature_range_degC' in results else None
        evaporation_mm = rp.open_water_evaporation_mm(temp, dtr, huss, wind_speed, results.index.dayofyear.values)
        results['evaporation_mm_day'] = np.where(valid_mask, evaporation_mm, np.nan)

        # Evaporation on the surface area at the measured volume
        results['r1_evap_ML_day'] = results['evaporation_mm_day'] * rp.surface_area_m2(
            reservoirs[0], results['r1_level_ML'].ffill().fillna(0).values) / 1e6
        results['r2_evap_ML_day'] = results['evaporation_mm_day'] * rp.surface_area_m2(
            reservoirs[1], results['r2_level_ML'].ffill().fillna(0).values) / 1e6
    
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
    col3.metric("Avg River→TND", f"{stats['avg_pump_river_r1']:.1f} ML/day")
    col4.metric("Avg TND→SCD", f"{stats['avg_pump_r1_r2']:.1f} ML/day")
    col5.metric("Avg SCD→Site", f"{stats['avg_pump_r2_site']:.1f} ML/day")

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
