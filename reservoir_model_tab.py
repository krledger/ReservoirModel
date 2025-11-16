"""
Tab 1: Model Simulation
Handles predictive reservoir simulation with climate scenarios
"""

import streamlit as st
import pandas as pd
from reservoir_operations import calculate_statistics, prepare_flow_scenario, prepare_prior_12_months_data
from reservoir_plotting import create_plots


def render_model_tab(system, reservoirs, pumps, min_year, max_year, presets, historical_results=None):
    """Render the model simulation tab"""
    
    st.header("Model Simulation")

    # Get parameters from sidebar (widgets automatically set session state via their keys)
    climate_scenario = st.session_state.get('climate_scenario', 'SSP1-26')
    drought_years = st.session_state.get('drought_years', [])
    rain_years = st.session_state.get('rain_years', [])
    start_year = st.session_state.get('model_start_year', presets['model_start_year'] if presets else 2015)
    end_year = st.session_state.get('model_end_year', presets['model_end_year'] if presets else 2025)
    demand = st.session_state.get('model_demand', presets['demand_avg'] if presets else 9.8)
    pump_river_to_r1_max = st.session_state.get('model_pump_river_r1', int(pumps[0]['max_rate_in_ML_day']))
    pump_r1_to_r2_max = st.session_state.get('model_pump_r1_r2', float(pumps[0]['max_rate_out_ML_day']))
    pump_r2_to_site_max = st.session_state.get('model_pump_r2_site', float(pumps[1]['max_rate_out_ML_day']))
    river_pump_low = st.session_state.get('model_river_pump_low', int(pumps[0]['cutoffs']['low_flow_ML_day']))
    river_pump_high = st.session_state.get('model_river_pump_high', int(pumps[0]['cutoffs']['high_flow_ML_day']))
    r1_initial = st.session_state.get('model_r1_initial', presets['r1_initial'] if presets else int(reservoirs[0]['initial_level_ML']))
    r2_initial = st.session_state.get('model_r2_initial', presets['r2_initial'] if presets else int(reservoirs[1]['initial_level_ML']))
    enable_random = st.session_state.get('enable_random', False)
    random_seed = st.session_state.get('random_seed', 42)

    # Create parameter signature to detect changes
    current_params = {
        'climate_scenario': climate_scenario,
        'drought_years': tuple(drought_years),
        'rain_years': tuple(rain_years),
        'start_year': start_year,
        'end_year': end_year,
        'demand': demand,
        'pump_river_to_r1_max': pump_river_to_r1_max,
        'pump_r1_to_r2_max': pump_r1_to_r2_max,
        'pump_r2_to_site_max': pump_r2_to_site_max,
        'river_pump_low': river_pump_low,
        'river_pump_high': river_pump_high,
        'r1_initial': r1_initial,
        'r2_initial': r2_initial,
        'enable_random': enable_random,
        'random_seed': random_seed
    }

    # Check if parameters changed or first run
    last_params = st.session_state.get('last_model_params', None)
    params_changed = (last_params != current_params)

    # Auto-run simulation on first load or when parameters change
    run_model = False

    # Auto-run simulation on first load if presets are available
    if not st.session_state.get('initial_run_complete', False):
        if presets:
            run_model = True
        else:
            return
    elif params_changed:
        run_model = True

    # MODEL SIMULATION LOGIC
    if run_model:
        try:
            # Reload climate data for selected scenario
            with st.spinner(f'Loading {climate_scenario} climate data...'):
                try:
                    system.load_climate_data(scenario=climate_scenario)
                except FileNotFoundError as e:
                    st.error(f"Climate data file not found: {e}")
                    return

            with st.spinner('Loading flow scenario...'):
                modified_flows = prepare_flow_scenario('.', drought_years, rain_years)
                system.flow_data = modified_flows

            params = {
                'demand_ML_day': demand,
                'pump_river_to_r1_max': pump_river_to_r1_max,
                'pump_r1_to_r2_max': pump_r1_to_r2_max,
                'pump_r2_to_site_max': pump_r2_to_site_max,
                'river_pump_low_cutoff': river_pump_low,
                'river_pump_high_cutoff': river_pump_high,
                'r1_initial': r1_initial,
                'r2_initial': r2_initial,
                'enable_random': enable_random,
                'random_seed': random_seed if enable_random else None
            }

            with st.spinner('Running simulation...'):
                start_date = f"{start_year}-01-01"
                end_date = f"{end_year}-12-31"
                results = system.simulate_reservoir_system(start_date, end_date, params)

                # Store in session state
                st.session_state.model_results = results
                st.session_state.initial_run_complete = True
                st.session_state.last_model_params = current_params

            r1_min = reservoirs[0].get('min_capacity_ML', 0)
            r2_min = reservoirs[1].get('min_capacity_ML', 0)

            # Calculate statistics
            stats = calculate_statistics(results, reservoirs, pumps)

            scenario_desc = f"{climate_scenario} - TYPICAL"
            if drought_years:
                scenario_desc += f" + {len(drought_years)} DROUGHT"
            if rain_years:
                scenario_desc += f" + {len(rain_years)} RAIN"
            if enable_random:
                scenario_desc += f" + RANDOM (seed={random_seed})"

            # Display statistics
            display_model_statistics(stats, len(results))

            # Prepare prior 12 months averages for horizontal reference lines
            prior_12m_avg = None
            if historical_results is not None:
                prior_12m_avg = prepare_prior_12_months_data(historical_results, results)

            # Create plots
            r1_turkeys = system.system_config['system']['reservoirs'][0]
            r2_surhs = system.system_config['system']['reservoirs'][1]

            fig = create_plots(results, scenario_desc, start_year, end_year, r1_min, r2_min,
                               river_pump_low, river_pump_high, r1_turkeys, r2_surhs,
                               results_prior_12m=prior_12m_avg)
            st.plotly_chart(fig, use_container_width=True)

            # Download results
            with st.expander("💾 Download Results"):
                csv = results.to_csv()
                st.download_button(
                    label="Download simulation results (CSV)",
                    data=csv,
                    file_name=f"reservoir_sim_{start_year}_{end_year}_{scenario_desc.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Simulation error: {e}")
            st.exception(e)

    elif st.session_state.get('model_results') is not None:
        # Display previously run results
        display_cached_results(system, reservoirs, pumps, climate_scenario, drought_years, rain_years,
                              start_year, end_year, river_pump_low, river_pump_high, enable_random, random_seed,
                              historical_results)
    else:
        st.info("Configure simulation parameters in the sidebar and the model will run automatically.")


def display_model_statistics(stats, num_days):
    """Display statistics for model simulation"""
    # Main metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Demand Deficit", f"{stats['total_deficit']:.1f} ML", f"{stats['deficit_days']} days")
    col2.metric("Avg Turkeys Nest Dam", f"{stats['avg_r1']:.1f} ML",
                f"{int(stats['days_of_storage_r1'])} days")
    col3.metric("Avg Surhs Creek Dam", f"{stats['avg_r2']:.1f} ML",
                f"{int(stats['days_of_storage_r2'])} days")
    col4.metric("Total Days of Storage", f"{int(stats['days_of_storage_total'])} days",
                help="Combined system storage at avg levels (demand + all losses, no inflows)")
    col5.metric("Min Surhs Creek Dam", f"{stats['min_r2']:.1f} ML")

    # Pump diagnostics
    st.markdown("### 🔧 Pump Activity Diagnostics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Days TND→SCD=0", f"{stats['days_pump_r1_r2_zero']}",
                f"{stats['days_pump_r1_r2_zero'] / num_days * 100:.1f}% of time")
    col2.metric("Days River→TND Active", f"{stats['days_pump_river_r1_active']}",
                f"{stats['days_pump_river_r1_active'] / num_days * 100:.1f}% of time")
    col3.metric("Avg TND→SCD Flow", f"{stats['avg_pump_r1_r2']:.2f} ML/day")
    col4.metric("Total Fluvial (SCD)", f"{stats['total_fluvial']:.0f} ML")
    col5.metric("Total Pluvial (SCD)", f"{stats['total_pluvial_r2']:.0f} ML")

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

    # Water balance breakdown
    if 'daily_consumption_breakdown' in stats:
        st.markdown("#### 💧 Average Daily Water Balance")
        col1, col2, col3, col4, col5 = st.columns(5)
        breakdown = stats['daily_consumption_breakdown']
        col1.metric("Demand", f"{breakdown['demand']:.2f} ML/day")
        col2.metric("TND Losses", f"{breakdown['r1_losses']:.2f} ML/day",
                   f"Seep: {breakdown['r1_seepage']:.2f} + Evap: {breakdown['r1_evap']:.2f}")
        col3.metric("SCD Losses", f"{breakdown['r2_losses']:.2f} ML/day",
                   f"Seep: {breakdown['r2_seepage']:.2f} + Evap: {breakdown['r2_evap']:.2f}")
        col4.metric("Total Daily Loss", f"{breakdown['total_loss']:.2f} ML/day")
        col5.metric("Total Available", f"{breakdown['total_available']:.1f} ML",
                   f"TND: {breakdown['r1_available']:.0f} + SCD: {breakdown['r2_available']:.0f}")


def display_cached_results(system, reservoirs, pumps, climate_scenario, drought_years, rain_years,
                           start_year, end_year, river_pump_low, river_pump_high, enable_random, random_seed,
                           historical_results=None):
    """Display previously computed results from session state"""
    results = st.session_state.model_results

    r1_min = reservoirs[0].get('min_capacity_ML', 0)
    r2_min = reservoirs[1].get('min_capacity_ML', 0)

    # Calculate statistics
    stats = calculate_statistics(results, reservoirs, pumps)

    scenario_desc = f"{climate_scenario} - TYPICAL"
    if drought_years:
        scenario_desc += f" + {len(drought_years)} DROUGHT"
    if rain_years:
        scenario_desc += f" + {len(rain_years)} RAIN"
    if enable_random:
        scenario_desc += f" + RANDOM (seed={random_seed})"

    # Display statistics
    display_model_statistics(stats, len(results))

    # Prepare prior 12 months averages for horizontal reference lines
    prior_12m_avg = None
    if historical_results is not None:
        prior_12m_avg = prepare_prior_12_months_data(historical_results, results)

    # Create plots
    r1_turkeys = system.system_config['system']['reservoirs'][0]
    r2_surhs = system.system_config['system']['reservoirs'][1]

    fig = create_plots(results, scenario_desc, start_year, end_year, r1_min, r2_min,
                       river_pump_low, river_pump_high, r1_turkeys, r2_surhs,
                       results_prior_12m=prior_12m_avg)
    st.plotly_chart(fig, use_container_width=True)

    # Download results
    with st.expander("💾 Download Results"):
        csv = results.to_csv()
        st.download_button(
            label="Download simulation results (CSV)",
            data=csv,
            file_name=f"reservoir_sim_{start_year}_{end_year}_{scenario_desc.replace(' ', '_')}.csv",
            mime="text/csv"
        )


def render_model_sidebar(pumps, reservoirs, min_year, max_year, presets):
    """Render sidebar controls for model tab"""
    st.sidebar.header("Model Simulation Controls")

    # Initialize session state with presets on first load (before widgets are created)
    if presets and not st.session_state.get('presets_initialized', False):
        # Set initial values directly without checking if keys exist
        # This prevents the "default value but also set via Session State API" warning
        st.session_state.presets_initialized = True

        # Store preset values for comparison but don't set widget defaults here
        st.session_state.preset_r1_initial = presets['r1_initial']
        st.session_state.preset_r2_initial = presets['r2_initial']
        st.session_state.preset_demand = presets['demand_avg']
        st.session_state.preset_start_year = presets['model_start_year']
        st.session_state.preset_end_year = presets['model_end_year']
    elif not presets:
        st.sidebar.warning("Load historical data to auto-populate settings")

    # Climate Scenario Selection
    with st.sidebar.expander("🌡️ Climate Scenario", expanded=True):
        climate_scenario = st.selectbox(
            "Select climate data source:",
            options=["SSP1-26", "SSP2-45", "SSP5-85", "AGCD"],
            index=0,
            help="SSP = Shared Socioeconomic Pathways (future projections), AGCD = Australian Gridded Climate Data (actual observations)",
            key="climate_scenario"
        )


    # Scenario Selection
    with st.sidebar.expander("🌧️ Flow Scenario", expanded=True):
        st.markdown("**Insert Drought Years**")
        st.caption("Bottom 10% by annual flow")
        drought_input = st.text_input(
            "Drought years (comma-separated)",
            placeholder="e.g., 2030, 2050, 2070",
            help="Hydrological year spans Nov (year-1) to Oct (year)",
            label_visibility="collapsed",
            key="model_drought"
        )

        st.markdown("**Insert Extreme Rain Years**")
        st.caption("Top 10% by annual flow")
        rain_input = st.text_input(
            "Rain years (comma-separated)",
            placeholder="e.g., 2040, 2060, 2080",
            help="Hydrological year spans Nov (year-1) to Oct (year)",
            label_visibility="collapsed",
            key="model_rain"
        )

    # Parse extreme year inputs
    drought_years = []
    if drought_input.strip():
        try:
            drought_years = [int(y.strip()) for y in drought_input.split(',')]
        except ValueError:
            st.sidebar.error("Invalid drought year format")

    rain_years = []
    if rain_input.strip():
        try:
            rain_years = [int(y.strip()) for y in rain_input.split(',')]
        except ValueError:
            st.sidebar.error("Invalid rain year format")

    # Store in separate keys (not widget keys)
    if 'drought_years' not in st.session_state:
        st.session_state.drought_years = []
    if 'rain_years' not in st.session_state:
        st.session_state.rain_years = []
    st.session_state.drought_years = drought_years
    st.session_state.rain_years = rain_years

    # Date range
    with st.sidebar.expander("📅 Date Range", expanded=True):
        default_start = presets['model_start_year'] if presets else 2015
        default_end = presets['model_end_year'] if presets else 2025

        # Only set value if key doesn't exist in session state (first run)
        start_value = st.session_state.get('model_start_year', default_start)
        end_value = st.session_state.get('model_end_year', default_end)

        st.number_input("Start Year", min_value=min_year, max_value=max_year, value=start_value,
                       key="model_start_year")
        st.number_input("End Year", min_value=min_year, max_value=max_year, value=end_value,
                       key="model_end_year")

    # Water demand
    with st.sidebar.expander("💧 Water Demand", expanded=False):
        default_demand = presets['demand_avg'] if presets else 9.8
        st.number_input("Demand (ML/day)", min_value=0.0, value=float(default_demand), step=0.1,
                       key="model_demand")

    # Pump settings
    with st.sidebar.expander("⚙️ Pump Settings", expanded=False):
        st.markdown("**Pump: River → TND (Turkeys Nest)**")
        st.number_input(
            "Max Rate (ML/day)",
            min_value=0,
            value=int(pumps[0]['max_rate_in_ML_day']),
            step=1,
            key="model_pump_river_r1"
        )

        st.markdown("**Pump: TND → SCD (Turkeys → Surhs)**")
        st.number_input(
            "Max Rate (ML/day)",
            min_value=0.0,
            value=float(pumps[0]['max_rate_out_ML_day']),
            step=0.1,
            key="model_pump_r1_r2"
        )

        st.markdown("**Pump: SCD → Site (Surhs → Demand)**")
        st.number_input(
            "Max Rate (ML/day)",
            min_value=0.0,
            value=float(pumps[1]['max_rate_out_ML_day']),
            step=0.1,
            key="model_pump_r2_site",
            help="Physical pump capacity limit"
        )

    # River flow cutoffs
    with st.sidebar.expander("🌊 River Flow Cutoffs (Pump River→TND)", expanded=False):
        st.number_input(
            "Low Cutoff (ML/day)",
            min_value=0,
            value=int(pumps[0]['cutoffs']['low_flow_ML_day']),
            step=1,
            help="Minimum river flow to allow pumping",
            key="model_river_pump_low"
        )
        st.number_input(
            "High Cutoff (ML/day)",
            min_value=0,
            value=int(pumps[0]['cutoffs']['high_flow_ML_day']),
            step=100,
            help="Maximum river flow to allow pumping",
            key="model_river_pump_high"
        )

    # Initial reservoir levels
    with st.sidebar.expander("🏞️ Initial Reservoir Levels", expanded=False):
        default_r1 = presets['r1_initial'] if presets else int(reservoirs[0]['initial_level_ML'])
        default_r2 = presets['r2_initial'] if presets else int(reservoirs[1]['initial_level_ML'])

        st.number_input(
            f"Turkeys Nest Dam (ML, max {reservoirs[0]['capacity_ML']})",
            min_value=0, max_value=int(reservoirs[0]['capacity_ML']),
            value=default_r1, step=10,
            key="model_r1_initial"
        )
        st.number_input(
            f"Surhs Creek Dam (ML, max {reservoirs[1]['capacity_ML']})",
            min_value=0, max_value=int(reservoirs[1]['capacity_ML']),
            value=default_r2, step=10,
            key="model_r2_initial"
        )

    # Random variations
    with st.sidebar.expander("🎲 Random Variations", expanded=False):
        enable_random = st.checkbox(
            "Enable random variations",
            value=False,
            help="Add realistic variability to demand (±20%) and pumps (±10%)",
            key="enable_random"
        )
        st.number_input(
            "Random seed",
            min_value=0,
            value=42,
            help="Set seed for reproducible results",
            key="random_seed",
            disabled=not enable_random
        )
        if enable_random:
            st.caption("Demand: ±20% | Pumps: ±10%")
            st.caption("Smoothed random walk (weekly correlation)")
