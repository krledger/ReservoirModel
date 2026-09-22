"""
Tab 1: Model Simulation
Handles predictive reservoir simulation with climate scenarios
"""

import streamlit as st
import pandas as pd
from reservoir_operations import calculate_statistics, prepare_prior_12_months_data, operating_profile_params
import ModelInputs as mi
import ReservoirPhysics as rp

PROFILE_LABELS = {'as_operated': 'As operated (calibrated to site practice)',
                  'potential': 'Potential (fill both dams, maximum pump rates)'}
from reservoir_plotting import create_plots, split_subplots, PLOTLY_CONFIG


def render_model_tab(system, reservoirs, pumps, min_year, max_year, presets, historical_results=None):
    """Render the model simulation tab"""
    
    st.header("Model Simulation")

    # Get parameters from sidebar (widgets automatically set session state via their keys)
    climate_scenario = st.session_state.get('climate_scenario', mi.DEFAULT_SCENARIO)
    flow_source = mi.FLOW_ACTUAL
    operating_profile = st.session_state.get('operating_profile', 'as_operated')
    start_year = st.session_state.get('start_year', presets['model_start_year'] if presets else 2015)
    end_year = st.session_state.get('end_year', presets['model_end_year'] if presets else 2025)
    uses = {u: st.session_state.get(f'model_{u}_demand', d) for u, d in demand_defaults(presets).items()}
    site_demand = sum(uses.values())
    wtp_demand = st.session_state.get('model_wtp_demand', round(presets['wtp_avg'], 1) if presets else 0.5)
    demand = site_demand + wtp_demand
    levels = operating_levels(system.system_config)
    river_pump_low, river_pump_high = levels['river_low'], levels['withdraw_above']
    pump_river_to_r1_max = st.session_state.get('model_pump_river_r1', int(pumps[0]['max_rate_in_ML_day']))
    pump_r1_to_r2_max = st.session_state.get('model_pump_r1_r2', float(pumps[0]['max_rate_out_ML_day']))
    pump_r2_to_site_max = st.session_state.get('model_pump_r2_site', float(pumps[1]['max_rate_out_ML_day']))
    r1_initial = st.session_state.get('model_r1_initial', presets['r1_initial'] if presets else int(reservoirs[0]['initial_level_ML']))
    r2_initial = st.session_state.get('model_r2_initial', presets['r2_initial'] if presets else int(reservoirs[1]['initial_level_ML']))
    enable_random = st.session_state.get('enable_random', False)
    random_seed = st.session_state.get('random_seed', 42)

    # Create parameter signature to detect changes
    current_params = {
        'climate_scenario': climate_scenario,
        'flow_source': flow_source,
        'operating_profile': operating_profile,
        'start_year': start_year,
        'end_year': end_year,
        'site_uses': tuple(sorted(uses.items())),
        'wtp_demand': wtp_demand,
        'levels': tuple(sorted(levels.items())),
        'pump_river_to_r1_max': pump_river_to_r1_max,
        'pump_r1_to_r2_max': pump_r1_to_r2_max,
        'pump_r2_to_site_max': pump_r2_to_site_max,
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

            system.flow_data = mi.load_flows(flow_source)

            params = {
                'demand_ML_day': demand,
                'wtp_demand_ML_day': wtp_demand,
                'pump_river_to_r1_max': pump_river_to_r1_max,
                'pump_r1_to_r2_max': pump_r1_to_r2_max,
                'pump_r2_to_site_max': pump_r2_to_site_max,
                'r1_initial': r1_initial,
                'r2_initial': r2_initial,
                'enable_random': enable_random,
                'random_seed': random_seed if enable_random else None
            }
            params.update(levels_to_params(levels, reservoirs))
            if operating_profile != 'as_operated':
                profile = operating_profile_params(system.system_config, operating_profile)
                for key in ('river_operations', 'transfer_operations'):
                    if key in profile:
                        profile[key] = {**params.get(key, {}), **profile[key]}
                params.update(profile)
                st.caption(f"{PROFILE_LABELS.get(operating_profile, operating_profile)}: pump rates and operating "
                           f"rules from operating_profiles in reservoir_system.json override the pump settings.")

            # Keep the run inside the years both inputs cover
            first_year, last_year = mi.year_range(climate_scenario)
            if start_year < first_year or end_year > last_year:
                st.warning(f"Inputs cover {first_year} to {last_year}. Run limited to those years.")
                start_year, end_year = max(start_year, first_year), min(end_year, last_year)

            with st.spinner('Running simulation...'):
                start_date = f"{start_year}-01-01"
                end_date = f"{end_year}-12-31"
                results = system.simulate_reservoir_system(start_date, end_date, params)

                # Store in session state
                st.session_state.model_results = results
                st.session_state.initial_run_complete = True
                st.session_state.last_model_params = current_params

            r1_min = params['r1_min_capacity']
            r2_min = params['r2_min_capacity']

            # Calculate statistics
            stats = calculate_statistics(results, reservoirs, pumps)

            scenario_desc = f"AGCD to {mi.last_actual_climate_day():%Y}, then {climate_scenario}"
            if operating_profile != 'as_operated':
                scenario_desc += f" - {operating_profile.upper()}"
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
                               results_prior_12m=prior_12m_avg, r2_reserve=params.get('r2_community_reserve'))
            st.session_state.model_view = {'stats': stats, 'figs': split_subplots(fig), 'scenario_desc': scenario_desc,
                                           'start_year': start_year, 'end_year': end_year, 'csv': None}
            show_charts(st.session_state.model_view)

            # Download results
            with st.expander("💾 Download Results"):
                csv = model_csv(results)
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
        display_cached_results(system, reservoirs, pumps, climate_scenario, start_year, end_year, river_pump_low, river_pump_high, enable_random, random_seed,
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
    if 'site_deficit' in stats:
        st.caption(f"Shortfall by use: site {stats['site_deficit']:.1f} ML over {stats['site_deficit_days']} days "
                   f"(stops below the community reserve), water treatment {stats['wtp_deficit']:.1f} ML over "
                   f"{stats['wtp_deficit_days']} days (stops below the minimum level for pumping).")

    # Pump diagnostics
    st.markdown("### 🔧 Pump Activity Diagnostics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Days TND→SCD=0", f"{stats['days_pump_r1_r2_zero']}",
                f"{stats['days_pump_r1_r2_zero'] / num_days * 100:.1f}% of time")
    col2.metric("Days River→TND Active", f"{stats['days_pump_river_r1_active']}",
                f"{stats['days_pump_river_r1_active'] / num_days * 100:.1f}% of time")
    col3.metric("Avg TND→SCD Flow", f"{stats['avg_pump_r1_r2']:.1f} ML/day")
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
        col1.metric("Demand", f"{breakdown['demand']:.1f} ML/day")
        col2.metric("TND Losses", f"{breakdown['r1_losses']:.1f} ML/day",
                   f"Net seep: {breakdown['r1_seepage']:.1f} + Evap: {breakdown['r1_evap']:.1f}")
        col3.metric("SCD Losses", f"{breakdown['r2_losses']:.1f} ML/day",
                   f"Net seep: {breakdown['r2_seepage']:.1f} + Evap: {breakdown['r2_evap']:.1f}")
        col4.metric("Total Daily Loss", f"{breakdown['total_loss']:.1f} ML/day")
        col5.metric("Total Available", f"{breakdown['total_available']:.1f} ML",
                   f"TND: {breakdown['r1_available']:.0f} + SCD: {breakdown['r2_available']:.0f}")


def reservoirs_config(pumps, reservoirs):
    """Rebuild the config shape level_defaults reads from the pieces the sidebar is given"""
    return {'system': {'pumps': pumps, 'reservoirs': reservoirs}}


def level_defaults(config):
    """Operating levels as configured in reservoir_system.json"""
    res, pumps = config['system']['reservoirs'], config['system']['pumps']
    ro, to = pumps[0].get('operations', {}), pumps[1].get('operations', {})
    t_seep, s_seep = rp.seepage_settings(res[0]), rp.seepage_settings(res[1])
    return {
        'river_low': float(pumps[0]['cutoffs']['low_flow_ML_day']),
        'withdraw_above': float(ro.get('withdraw_above_ML_day') or pumps[0]['cutoffs']['high_flow_ML_day']),
        'reinstate_below': float(ro.get('reinstate_below_ML_day') or ro.get('withdraw_above_ML_day')
                                 or pumps[0]['cutoffs']['high_flow_ML_day']),
        'reinstate_after': int(ro.get('reinstate_after_days', 0)),
        'tnd_stop': float(ro.get('tnd_stop_pct') or 100),
        'tnd_restart': float(ro.get('tnd_restart_pct') or ro.get('tnd_stop_pct') or 100),
        'scd_target': float(to.get('scd_target_pct', 70)),
        'scd_stop': float(to.get('scd_stop_pct') or 100),
        'tnd_min_pct': round(float(res[0].get('min_capacity_ML', 0)) / res[0]['capacity_ML'] * 100, 1),
        'scd_min_pct': round(float(res[1].get('min_capacity_ML', 0)) / res[1]['capacity_ML'] * 100, 1),
        'scd_reserve_pct': float(res[1].get('community_reserve_pct', 0)),
        'offtakes': round(float(to.get('offtakes_ML_day', 0.0)), 1),
        'tnd_seep_gross': float(t_seep['gross_ML_day']),
        'tnd_seep_returned': float(t_seep['returned_pct']),
        'scd_seep_gross': float(s_seep['gross_ML_day']),
        'scd_seep_returned': float(s_seep['returned_pct']),
    }


DEMAND_LABELS = {
    'plant': ("Processing plant", "Meter 109, raw water into the plant tank"),
    'gland': ("Gland water", "Meters 106 and 167"),
    'dust': ("Dust suppression (water carts)", "Meters 032 and 067, SAR and NOL water cart tanks"),
    'minor': ("Minor raw users", "Meters 019, 033, 118, 161 and 162: golf and SPQ, HV workshop, school, "
                                 "LV washdown and Simmco"),
    'other': ("Other (unmetered)", "Meter 014 less all metered uses: unmetered draw and meter error"),
}
DEMAND_FALLBACK = {'plant': 3.7, 'gland': 1.1, 'dust': 1.6, 'minor': 0.2, 'other': 0.1}


def demand_defaults(presets):
    """Site demand by use, ML/day, one decimal place"""
    return {u: round(float(presets.get(f'{u}_avg') if presets and presets.get(f'{u}_avg') is not None else v), 1)
            for u, v in DEMAND_FALLBACK.items()}


def starting_levels(start, reservoirs, actuals):
    """Dam volumes at the model start: the last actual reading on or before the start date, or the
    initial levels in reservoir_system.json when the start is before the actual record."""
    json_levels = (int(reservoirs[0]['initial_level_ML']), int(reservoirs[1]['initial_level_ML']))
    if actuals is None:
        return (*json_levels, "Starting levels from reservoir_system.json.")
    a = actuals[['r1_level_ML', 'r2_level_ML']].dropna()
    if a.empty or start < a.index.min():
        return (*json_levels, f"Start is before the actual record ({a.index.min():%d/%m/%Y}); starting levels "
                              f"from reservoir_system.json.")
    row = a[a.index <= start].iloc[-1]
    return (int(round(row['r1_level_ML'])), int(round(row['r2_level_ML'])),
            f"Starting levels from the actual reading of {row.name:%d/%m/%Y}.")


def d_net(reservoir, gross, returned_pct, pct_full=80):
    """Net seepage at a given fill, for the sidebar note"""
    s = {**rp.seepage_settings(reservoir), 'gross_ML_day': gross, 'returned_pct': returned_pct}
    return float(rp.seepage_ML_day(reservoir, reservoir['capacity_ML'] * pct_full / 100, s)[2])


def operating_levels(config):
    """Current sidebar values, falling back to the configuration"""
    d = level_defaults(config)
    return {k: st.session_state.get(f'lvl_{k}', v) for k, v in d.items()}


def levels_to_params(lv, reservoirs):
    """Sidebar operating levels as simulation params (override reservoir_system.json for this run)"""
    cap_t, cap_s = reservoirs[0]['capacity_ML'], reservoirs[1]['capacity_ML']
    return {
        'river_pump_low_cutoff': lv['river_low'],
        'river_pump_high_cutoff': lv['withdraw_above'],
        'r1_min_capacity': lv['tnd_min_pct'] / 100 * cap_t,
        'r2_min_capacity': lv['scd_min_pct'] / 100 * cap_s,
        'r2_community_reserve': lv['scd_reserve_pct'] / 100 * cap_s,
        'river_operations': {
            'withdraw_above_ML_day': lv['withdraw_above'],
            'reinstate_below_ML_day': lv['reinstate_below'],
            'reinstate_after_days': int(lv['reinstate_after']),
            'tnd_stop_pct': lv['tnd_stop'],
            'tnd_restart_pct': lv['tnd_restart'],
        },
        'transfer_operations': {
            'scd_target_pct': lv['scd_target'],
            'scd_stop_pct': lv['scd_stop'],
            'offtakes_ML_day': lv['offtakes'],
        },
        'seepage': {
            'r1': {'gross_ML_day': lv['tnd_seep_gross'], 'returned_pct': lv['tnd_seep_returned']},
            'r2': {'gross_ML_day': lv['scd_seep_gross'], 'returned_pct': lv['scd_seep_returned']},
        },
    }


def show_charts(view):
    """One chart per panel, each with its own toolbar and PNG export"""
    st.markdown(f"**Reservoir System Analysis - {view['scenario_desc']} ({view['start_year']}-{view['end_year']})**")
    figs = view.get('figs')
    if figs is None and view.get('fig') is not None:
        figs = view['figs'] = split_subplots(view['fig'])
    for i, f in enumerate(figs or []):
        st.plotly_chart(f, config=PLOTLY_CONFIG, key=f"model_chart_{i}")


def model_csv(results):
    """CSV of the current results, built once per run rather than on every rerun."""
    view = st.session_state.get('model_view')
    if view is not None and view.get('csv') is not None:
        return view['csv']
    csv = results.to_csv()
    if view is not None:
        view['csv'] = csv
    return csv


def display_cached_results(system, reservoirs, pumps, climate_scenario, start_year, end_year, river_pump_low, river_pump_high, enable_random, random_seed,
                           historical_results=None):
    """Show the last run without recomputing statistics or rebuilding the figure"""
    results = st.session_state.model_results
    view = st.session_state.get('model_view')
    if view is None:
        stats = calculate_statistics(results, reservoirs, pumps)
        fig = create_plots(results, climate_scenario, start_year, end_year,
                           reservoirs[0].get('min_capacity_ML', 0), reservoirs[1].get('min_capacity_ML', 0),
                           river_pump_low, river_pump_high, reservoirs[0], reservoirs[1])
        view = {'stats': stats, 'figs': split_subplots(fig), 'scenario_desc': climate_scenario,
                'start_year': start_year, 'end_year': end_year, 'csv': None}
        st.session_state.model_view = view

    display_model_statistics(view['stats'], len(results))
    show_charts(view)

    with st.expander("💾 Download Results"):
        st.download_button(
            label="Download simulation results (CSV)",
            data=model_csv(results),
            file_name=f"reservoir_sim_{view['start_year']}_{view['end_year']}_"
                      f"{view['scenario_desc'].replace(' ', '_').replace(',', '')}.csv",
            mime="text/csv"
        )


def render_model_sidebar(pumps, reservoirs, min_year, max_year, presets, historical_results=None):
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
    with st.sidebar.expander("🌡️ Climate and Operation", expanded=True):
        st.selectbox(
            f"Climate scenario after {mi.last_actual_climate_day():%b %Y}:",
            options=mi.CLIMATE_SOURCES,
            index=mi.CLIMATE_SOURCES.index(mi.DEFAULT_SCENARIO),
            help="AGCD observations are used to the last actual day; the SSP carries on from there, "
                 "baselined to AGCD so there is no step at the join.  River flow is always the actual "
                 "Sellheim record, then the bootstrap scenario.",
            key="climate_scenario"
        )
        st.selectbox(
            "Operating profile:",
            options=list(PROFILE_LABELS),
            format_func=lambda k: PROFILE_LABELS[k],
            index=0,
            key="operating_profile"
        )


    # Water demand
    with st.sidebar.expander("💧 Water Demand", expanded=False):
        d = level_defaults(reservoirs_config(pumps, reservoirs))
        st.caption("ML/day.  Defaults are the last 12 months of Weekly SWB.")
        st.markdown("**Site (stops below the community reserve)**")
        for use, default in demand_defaults(presets).items():
            label, help_text = DEMAND_LABELS[use]
            st.number_input(label, min_value=0.0, value=default, step=0.1, format="%.1f",
                            key=f"model_{use}_demand", help=help_text)
        site = sum(st.session_state.get(f'model_{u}_demand', v) for u, v in demand_defaults(presets).items())
        st.caption(f"Site total: {site:.1f} ML/day")
        st.markdown("**Community (continues to the minimum level for pumping)**")
        wtp_default = round(float(presets['wtp_avg']), 1) if presets else 0.5
        st.number_input("Water treatment", min_value=0.0, value=wtp_default, step=0.1, format="%.1f",
                        key="model_wtp_demand",
                        help="Meter 166, WTP raw in, supplied out of 014.  Supplies the town and site potable water.")
        total = site + st.session_state.get('model_wtp_demand', wtp_default)
        st.caption(f"Drawn from SCD (meter 014): {total:.1f} ML/day")
        st.markdown("**Taken before SCD**")
        st.number_input("Station offtakes, Carse O Gowrie and Kirkton", min_value=0.0,
                        value=round(d['offtakes'], 1), step=0.1, format="%.1f", key="lvl_offtakes",
                        help="Meters 044 and 045, taken from the TND to SCD main")

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

    # Operating levels
    with st.sidebar.expander("🎚️ Operating Levels and Cutoffs", expanded=False):
        d = level_defaults(reservoirs_config(pumps, reservoirs))
        cap_t, cap_s = reservoirs[0]['capacity_ML'], reservoirs[1]['capacity_ML']
        st.caption("Defaults from reservoir_system.json.  Changes apply to the model run only.")
        st.markdown("**River pumps (Sellheim ML/day)**")
        st.number_input("Low flow cutoff", min_value=0.0, value=d['river_low'], step=10.0, key="lvl_river_low",
                        help="No pumping below this flow")
        st.number_input("Withdraw pumps above", min_value=0.0, value=d['withdraw_above'], step=1000.0,
                        key="lvl_withdraw_above", help="Pumps are lifted out of the river above this flow")
        st.number_input("Reinstate below", min_value=0.0, value=d['reinstate_below'], step=1000.0,
                        key="lvl_reinstate_below")
        st.number_input("Days below before reinstating", min_value=0, value=d['reinstate_after'], step=1,
                        key="lvl_reinstate_after",
                        help="Time to put the pumps back after a flood: flow must stay below the reinstatement "
                             "level for this many days.  Site record 0 to 6 days; fitted value 7.")
        st.markdown("**Turkeys Nest (% of capacity)**")
        st.number_input("Stop filling at", min_value=0.0, max_value=100.0, value=d['tnd_stop'], step=1.0,
                        key="lvl_tnd_stop")
        st.number_input("Restart filling at", min_value=0.0, max_value=100.0, value=d['tnd_restart'], step=1.0,
                        key="lvl_tnd_restart")
        tnd_min = st.number_input("Minimum", min_value=0.0, max_value=100.0, value=d['tnd_min_pct'], step=0.5,
                                  format="%.1f", key="lvl_tnd_min_pct")
        st.caption(f"Below minimum level for pumping {tnd_min / 100 * cap_t:,.0f} ML")
        st.markdown("**Surhs Creek (% of capacity)**")
        st.number_input("Operating level (booster restarts)", min_value=0.0, max_value=100.0,
                        value=d['scd_target'], step=1.0, key="lvl_scd_target")
        st.number_input("Booster stops at", min_value=0.0, max_value=100.0, value=d['scd_stop'], step=1.0,
                        key="lvl_scd_stop")
        reserve = st.number_input("Community reserve", min_value=0.0, max_value=100.0,
                                  value=d['scd_reserve_pct'], step=1.0, format="%.0f", key="lvl_scd_reserve_pct",
                                  help="Legislated at 20 per cent.  Below this level the mine stops and SCD "
                                       "supplies only the water treatment plant.")
        st.caption(f"Legislated.  Mine supply stops below {reserve / 100 * cap_s:,.0f} ML")
        scd_min = st.number_input("Minimum", min_value=0.0, max_value=100.0, value=d['scd_min_pct'], step=0.5,
                                  format="%.1f", key="lvl_scd_min_pct")
        st.caption(f"Below minimum level for pumping {scd_min / 100 * cap_s:,.0f} ML")

    # Seepage
    with st.sidebar.expander("🪨 Seepage", expanded=False):
        d = level_defaults(reservoirs_config(pumps, reservoirs))
        st.caption("Gross seepage at full supply (ML/day) falls in proportion to the volume stored.  Returned is "
                   "the share recovered and pumped back to the same dam.  Defaults from reservoir_system.json.")
        st.number_input("Turkeys Nest gross", min_value=0.0, value=d['tnd_seep_gross'], step=0.1, format="%.1f",
                        key="lvl_tnd_seep_gross")
        st.number_input("Turkeys Nest returned (%)", min_value=0.0, max_value=100.0, value=d['tnd_seep_returned'],
                        step=5.0, format="%.0f", key="lvl_tnd_seep_returned")
        st.number_input("Surhs Creek gross", min_value=0.0, value=d['scd_seep_gross'], step=0.1, format="%.1f",
                        key="lvl_scd_seep_gross")
        st.number_input("Surhs Creek returned (%)", min_value=0.0, max_value=100.0, value=d['scd_seep_returned'],
                        step=5.0, format="%.0f", key="lvl_scd_seep_returned",
                        help="Recovery on meters 081, 092 and 093 pumped back to SCD")
        net_t = d_net(reservoirs[0], st.session_state.get('lvl_tnd_seep_gross', d['tnd_seep_gross']),
                      st.session_state.get('lvl_tnd_seep_returned', d['tnd_seep_returned']))
        net_s = d_net(reservoirs[1], st.session_state.get('lvl_scd_seep_gross', d['scd_seep_gross']),
                      st.session_state.get('lvl_scd_seep_returned', d['scd_seep_returned']))
        st.caption(f"Net seepage at 80% full: TND {net_t:.1f}, SCD {net_s:.1f} ML/day")

    # Initial reservoir levels
    with st.sidebar.expander("🏞️ Initial Reservoir Levels", expanded=False):
        start = pd.Timestamp(f"{st.session_state.get('start_year', presets['model_start_year'] if presets else 2015)}-01-01")
        default_r1, default_r2, basis = starting_levels(start, reservoirs, historical_results)
        # Reset the inputs to the new defaults when the start year changes
        if st.session_state.get('initial_levels_for') != start:
            st.session_state.initial_levels_for = start
            st.session_state.model_r1_initial = default_r1
            st.session_state.model_r2_initial = default_r2
        st.caption(basis)

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
