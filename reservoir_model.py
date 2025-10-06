import streamlit as st
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Reservoir System Simulation", layout="wide")


# ============================================================================
# SCENARIO UTILITIES
# ============================================================================

def find_scenario_files(data_dir):
    """Scan directory for scenario parquet files"""
    scenarios = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        return scenarios

    for fpath in data_path.glob("*.parquet"):
        fname = fpath.name

        if "typical" in fname.lower():
            scenarios["TYPICAL"] = str(fpath)
        elif "drought" in fname.lower() and "extreme" in fname.lower():
            scenarios["DROUGHT_TEMPLATE"] = str(fpath)
        elif "rain" in fname.lower() and "extreme" in fname.lower():
            scenarios["RAIN_TEMPLATE"] = str(fpath)

    return scenarios


def load_extreme_template(template_path):
    """Load 365-day extreme template"""
    df = pd.read_parquet(template_path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    return df


def insert_extreme_year_into_flows(base_flows, extreme_template, target_year, scenario_type):
    """
    Insert extreme year template into target hydrological year.
    Hydrological year: Nov (target_year-1) through Oct (target_year)
    """
    flows = base_flows.copy()

    # Map template dates to target hydrological year
    extreme_data = extreme_template.copy()

    def shift_date(d):
        if d.month >= 11:  # Nov, Dec
            return d.replace(year=target_year - 1)
        else:  # Jan-Oct
            return d.replace(year=target_year)

    extreme_data.index = extreme_data.index.map(shift_date)

    # Define removal range
    hydro_start = pd.Timestamp(f"{target_year - 1}-11-01")
    hydro_end = pd.Timestamp(f"{target_year}-10-31")

    # Remove target hydrological year
    flows_filtered = flows[(flows.index < hydro_start) | (flows.index > hydro_end)]

    # Combine
    result = pd.concat([flows_filtered, extreme_data]).sort_index()

    return result


# ============================================================================
# RESERVOIR SYSTEM CLASS
# ============================================================================

class ReservoirSystem:
    """Main class to handle reservoir system simulation"""

    def __init__(self, base_path='.'):
        self.base_path = Path(base_path)
        self.climate_data = None
        self.flow_data = None
        self.system_config = None

    def load_system_config(self, config_path='reservoir_system.json'):
        """Load reservoir system configuration"""
        config_file = self.base_path / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"System configuration not found: {config_file}")

        with open(config_file, 'r') as f:
            self.system_config = json.load(f)
        return self.system_config

    def load_climate_data(self, scenario='SSP1-26'):
        """Load climate data from parquet file"""
        climate_base = self.base_path / 'metricsDataFiles'

        if not climate_base.exists():
            raise FileNotFoundError(f"Climate data directory not found: {climate_base}")

        parquet_file = None
        for root, dirs, files in os.walk(climate_base):
            if 'raw_daily.parquet' in files:
                parquet_file = Path(root) / 'raw_daily.parquet'
                break

        if not parquet_file or not parquet_file.exists():
            raise FileNotFoundError(f"Climate data file 'raw_daily.parquet' not found in {climate_base}")

        self.climate_data = pd.read_parquet(parquet_file)
        self.climate_data.index = pd.to_datetime(self.climate_data.index)
        return self.climate_data

    def load_flow_data(self, flow_file='flows_bootstrap_typical.parquet'):
        """Load river flow data from parquet file"""
        flow_base = self.base_path / 'wmipData'

        if not flow_base.exists():
            raise FileNotFoundError(f"Flow data directory not found: {flow_base}")

        parquet_file = flow_base / flow_file

        if not parquet_file.exists():
            raise FileNotFoundError(f"Flow data file not found: {parquet_file}")

        self.flow_data = pd.read_parquet(parquet_file)

        if 'Date' in self.flow_data.columns:
            self.flow_data['Date'] = pd.to_datetime(self.flow_data['Date'])
            self.flow_data = self.flow_data.set_index('Date')
        elif not isinstance(self.flow_data.index, pd.DatetimeIndex):
            self.flow_data.index = pd.to_datetime(self.flow_data.index)

        return self.flow_data

    def calculate_relative_humidity(self, temperature, specific_humidity):
        """Calculate relative humidity (%) from temperature and specific humidity"""
        es = 6.108 * np.exp((17.27 * temperature) / (temperature + 237.3))
        p = 1013.25
        e = (specific_humidity / 1000) * p / (0.622 + 0.378 * (specific_humidity / 1000))
        rh = (e / es) * 100
        rh = np.clip(rh, 0, 100)
        return rh

    def calculate_evaporation(self, temperature, humidity, wind_speed=2.0):
        """Calculate evaporation using Penman equation (simplified)"""
        es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
        ea = (humidity / 1000) * 101.325 / 0.622
        vpd = es - ea
        vpd = np.maximum(vpd, 0)
        evaporation = 0.5 * (temperature / 20) * vpd * (1 + 0.5 * wind_speed / 2)
        return np.maximum(evaporation, 0)

    def simulate_reservoir_system(self, start_date, end_date, params):
        """Simulate the reservoir system with custom parameters"""
        # Filter data
        mask = (self.climate_data.index >= start_date) & (self.climate_data.index <= end_date)
        climate = self.climate_data.loc[mask].copy()

        mask = (self.flow_data.index >= start_date) & (self.flow_data.index <= end_date)
        flows = self.flow_data.loc[mask].copy()

        data = climate.join(flows, how='inner', rsuffix='_flow')

        if len(data) == 0:
            raise ValueError("No overlapping data after join")

        # Extract variables
        temperature = data['tas_Ravenswood_degC'].values
        specific_humidity = data['huss_Ravenswood_g_per_kg'].values
        precipitation = data['pr_Ravenswood_mm_day'].values

        # Try to get wind speed
        wind = None
        for wind_col in ['sfcWind_Ravenswood_ms', 'sfcwind_Ravenswood_ms', 'wind_Ravenswood_ms']:
            if wind_col in data.columns:
                wind = data[wind_col].values
                break

        if wind is None:
            wind = np.ones(len(data)) * 2.0

        if 'ML_day' in data.columns:
            river_flow = data['ML_day'].values
        elif 'CUMECS' in data.columns:
            river_flow = data['CUMECS'].values * 86.4
        else:
            river_flow = np.ones(len(data)) * 100

        relative_humidity = self.calculate_relative_humidity(temperature, specific_humidity)
        evaporation_mm = self.calculate_evaporation(temperature, specific_humidity, wind)

        # Get configuration
        reservoirs = self.system_config['system']['reservoirs']
        pumps = self.system_config['system']['pumps']
        turkeys_nest = reservoirs[0]
        surhs_creek = reservoirs[1]

        turkeys_min_capacity = turkeys_nest.get('min_capacity_ML', 0)
        surhs_min_capacity = surhs_creek.get('min_capacity_ML', 0)

        # Initialize arrays
        n = len(data)
        turkeys_level = np.zeros(n)
        surhs_level = np.zeros(n)
        pump1_flow = np.zeros(n)
        pump2_flow = np.zeros(n)
        pump3_flow = np.zeros(n)
        fluvial_inflow = np.zeros(n)
        pluvial_inflow_turkeys = np.zeros(n)
        pluvial_inflow_surhs = np.zeros(n)
        total_inflow = np.zeros(n)
        turkeys_evap_loss = np.zeros(n)
        surhs_evap_loss = np.zeros(n)
        demand_supplied = np.zeros(n)
        demand_deficit = np.zeros(n)

        # Initial levels
        turkeys_level[0] = params.get('turkeys_initial', turkeys_nest['initial_level_ML'])
        surhs_level[0] = params.get('surhs_initial', surhs_creek['initial_level_ML'])

        # Parameters
        pump1_max_in = params.get('pump1_max_in_rate', pumps[0]['max_rate_in_ML_day'])
        pump1_max_out = params.get('pump1_max_out_rate', pumps[0]['max_rate_out_ML_day'])
        pump2_max_out = params.get('pump2_max_out_rate', pumps[1]['max_rate_out_ML_day'])
        low_cutoff = params.get('pump1_low_cutoff', pumps[0]['cutoffs']['low_flow_ML_day'])
        high_cutoff = params.get('pump1_high_cutoff', pumps[0]['cutoffs']['high_flow_ML_day'])
        demand_ML_day = params.get('demand_ML_day', 9.8)

        # Simulation loop
        for i in range(1, n):
            tn_level = turkeys_level[i - 1]
            sc_level = surhs_level[i - 1]

            # Pump 1: River to Turkeys Nest
            available_capacity_tn = turkeys_nest['capacity_ML'] - tn_level
            if river_flow[i] >= low_cutoff and river_flow[i] <= high_cutoff and available_capacity_tn > 0:
                pump1 = min(pump1_max_in, available_capacity_tn)
            else:
                pump1 = 0
            pump1_flow[i] = pump1

            # Fluvial inflow
            if precipitation[i] >= surhs_creek['inflows']['fluvial']['min_rain_mm']:
                fluvial = precipitation[i] * surhs_creek['inflows']['fluvial']['coefficient_ML_per_mm']
            else:
                fluvial = 0
            fluvial_inflow[i] = fluvial

            # Pluvial inflow
            if precipitation[i] >= 2.0:
                pluvial_turkeys = (precipitation[i] / 1000) * turkeys_nest['surface_area_m²'] / 1000
                pluvial_surhs = (precipitation[i] / 1000) * surhs_creek['surface_area_m²'] / 1000
            else:
                pluvial_turkeys = 0
                pluvial_surhs = 0

            pluvial_inflow_turkeys[i] = pluvial_turkeys
            pluvial_inflow_surhs[i] = pluvial_surhs
            total_inflow[i] = fluvial + pluvial_surhs

            # Evaporation
            tn_evap = (evaporation_mm[i] / 1000) * turkeys_nest['surface_area_m²'] / 1000
            sc_evap = (evaporation_mm[i] / 1000) * surhs_creek['surface_area_m²'] / 1000
            turkeys_evap_loss[i] = tn_evap
            surhs_evap_loss[i] = sc_evap

            # Update Turkeys Nest
            tn_level += pump1 + pluvial_turkeys
            tn_level -= turkeys_nest['losses']['seepage_ML_day'] + tn_evap

            # Pump 2: Turkeys Nest to Surhs Creek
            available_water_tn = max(0, tn_level - turkeys_min_capacity)
            pump2 = min(pump1_max_out, available_water_tn)
            pump2 = max(0, pump2)
            pump2_flow[i] = pump2
            tn_level -= pump2
            tn_level = max(turkeys_min_capacity, min(tn_level, turkeys_nest['capacity_ML']))
            turkeys_level[i] = tn_level

            # Update Surhs Creek
            sc_level += pump2 + fluvial + pluvial_surhs
            sc_level -= surhs_creek['losses']['seepage_ML_day'] + sc_evap

            # Pump 3: Demand supply
            available_water_sc = max(0, sc_level - surhs_min_capacity)
            pump3 = min(demand_ML_day, available_water_sc)
            pump3 = max(0, pump3)
            pump3_flow[i] = pump3
            sc_level -= pump3

            demand_supplied[i] = pump3
            demand_deficit[i] = demand_ML_day - pump3

            sc_level = max(surhs_min_capacity, min(sc_level, surhs_creek['capacity_ML']))
            surhs_level[i] = sc_level

        # Create results dataframe
        results = pd.DataFrame({
            'date': data.index,
            'river_flow_ML_day': river_flow,
            'temperature_degC': temperature,
            'specific_humidity_g_kg': specific_humidity,
            'relative_humidity_pct': relative_humidity,
            'precipitation_mm_day': precipitation,
            'evaporation_mm_day': evaporation_mm,
            'fluvial_inflow_ML': fluvial_inflow,
            'pluvial_inflow_turkeys_ML': pluvial_inflow_turkeys,
            'pluvial_inflow_surhs_ML': pluvial_inflow_surhs,
            'total_inflow_ML': total_inflow,
            'turkeys_evap_ML_day': turkeys_evap_loss,
            'surhs_evap_ML_day': surhs_evap_loss,
            'pump1_flow_ML_day': pump1_flow,
            'pump2_flow_ML_day': pump2_flow,
            'pump3_flow_ML_day': pump3_flow,
            'turkeys_nest_level_ML': turkeys_level,
            'surhs_creek_level_ML': surhs_level,
            'demand_supplied_ML': demand_supplied,
            'demand_deficit_ML': demand_deficit
        })
        results.set_index('date', inplace=True)
        return results


# ============================================================================
# STREAMLIT APP
# ============================================================================

@st.cache_resource
def load_system():
    """Load system config and climate data once"""
    system = ReservoirSystem(base_path='.')
    system.load_system_config()
    system.load_climate_data()
    return system


def prepare_flow_scenario(base_path, drought_years, rain_years):
    """Load and modify flow scenario based on user selections (not cached)"""
    # Load base TYPICAL scenario
    flow_base = Path(base_path) / 'wmipData'
    typical_file = flow_base / 'flows_bootstrap_typical.parquet'

    if not typical_file.exists():
        raise FileNotFoundError(f"TYPICAL scenario not found: {typical_file}")

    base_flows = pd.read_parquet(typical_file)

    if 'Date' in base_flows.columns:
        base_flows['Date'] = pd.to_datetime(base_flows['Date'])
        base_flows = base_flows.set_index('Date')

    # Find available templates
    scenarios = find_scenario_files(flow_base)

    # Apply extreme year insertions
    if drought_years and 'DROUGHT_TEMPLATE' in scenarios:
        drought_template = load_extreme_template(scenarios['DROUGHT_TEMPLATE'])
        for year in drought_years:
            base_flows = insert_extreme_year_into_flows(
                base_flows, drought_template, year, 'DROUGHT'
            )

    if rain_years and 'RAIN_TEMPLATE' in scenarios:
        rain_template = load_extreme_template(scenarios['RAIN_TEMPLATE'])
        for year in rain_years:
            base_flows = insert_extreme_year_into_flows(
                base_flows, rain_template, year, 'EXTREME_RAIN'
            )

    return base_flows


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
    st.error(f"Error loading data: {e}")
    st.stop()

# Title
st.title("🌊 Reservoir System Simulation Dashboard")

# Sidebar controls
st.sidebar.header("Simulation Controls")

# Scenario Selection
with st.sidebar.expander("🎭 Flow Scenario", expanded=True):
    st.info("Base: TYPICAL (middle 80% of years)")

    st.markdown("**Insert Drought Years**")
    st.caption("Bottom 10% by annual flow")
    drought_input = st.text_input(
        "Drought years (comma-separated)",
        placeholder="e.g., 2030, 2050, 2070",
        help="Hydrological year spans Nov (year-1) to Oct (year)",
        label_visibility="collapsed"
    )

    st.markdown("**Insert Extreme Rain Years**")
    st.caption("Top 10% by annual flow")
    rain_input = st.text_input(
        "Rain years (comma-separated)",
        placeholder="e.g., 2040, 2060, 2080",
        help="Hydrological year spans Nov (year-1) to Oct (year)",
        label_visibility="collapsed"
    )

# Parse extreme year inputs
drought_years = []
if drought_input.strip():
    try:
        drought_years = [int(y.strip()) for y in drought_input.split(',')]
        st.sidebar.success(f"✓ {len(drought_years)} drought year(s)")
    except:
        st.sidebar.error("Invalid drought years format")

rain_years = []
if rain_input.strip():
    try:
        rain_years = [int(y.strip()) for y in rain_input.split(',')]
        st.sidebar.success(f"✓ {len(rain_years)} rain year(s)")
    except:
        st.sidebar.error("Invalid rain years format")

with st.sidebar.expander("📅 Date Range", expanded=True):
    start_year = st.number_input("Start Year", min_value=min_year, max_value=max_year, value=2015)
    end_year = st.number_input("End Year", min_value=min_year, max_value=max_year, value=2025)

with st.sidebar.expander("💧 Water Demand", expanded=False):
    demand = st.number_input("Demand (ML/day)", min_value=0.0, value=9.8, step=0.1)

with st.sidebar.expander("⚙️ Pump Settings", expanded=False):
    pump1_max_in = st.number_input("Pump 1 Max In (ML/day)", min_value=0, value=int(pumps[0]['max_rate_in_ML_day']),
                                   step=1)
    pump1_max_out = st.number_input("Pump 1 Max Out (ML/day)", min_value=0.0,
                                    value=float(pumps[0]['max_rate_out_ML_day']), step=0.1)

with st.sidebar.expander("🎯 Flow Cutoffs", expanded=False):
    pump1_low = st.number_input("Pump 1 Low Cutoff (ML/day)", min_value=0,
                                value=int(pumps[0]['cutoffs']['low_flow_ML_day']), step=1)
    pump1_high = st.number_input("Pump 1 High Cutoff (ML/day)", min_value=0,
                                 value=int(pumps[0]['cutoffs']['high_flow_ML_day']), step=100)

with st.sidebar.expander("🏞️ Initial Reservoir Levels", expanded=False):
    turkeys_initial = st.number_input(
        f"Turkeys Nest (ML, max {reservoirs[0]['capacity_ML']})",
        min_value=0, max_value=int(reservoirs[0]['capacity_ML']),
        value=int(reservoirs[0]['initial_level_ML']), step=10
    )
    surhs_initial = st.number_input(
        f"Surhs Creek (ML, max {reservoirs[1]['capacity_ML']})",
        min_value=0, max_value=int(reservoirs[1]['capacity_ML']),
        value=int(reservoirs[1]['initial_level_ML']), step=10
    )

with st.sidebar.expander("📊 Chart Display Options", expanded=False):
    show_reservoirs = st.checkbox("Reservoir Levels", value=True)
    show_flows = st.checkbox("River Flow", value=True)
    show_climate = st.checkbox("Climate (Temp/Humidity/Precip)", value=True)
    show_evap = st.checkbox("Evaporation", value=False)
    show_inflow = st.checkbox("Inflow Components", value=False)

run_simulation = st.sidebar.button("▶ Run Simulation", type="primary", use_container_width=True)

# Main content
if run_simulation:
    try:
        # Prepare flow scenario with modifications
        with st.spinner('Loading flow scenario...'):
            modified_flows = prepare_flow_scenario('.', drought_years, rain_years)
            system.flow_data = modified_flows

        params = {
            'demand_ML_day': demand,
            'pump1_max_in_rate': pump1_max_in,
            'pump1_max_out_rate': pump1_max_out,
            'pump1_low_cutoff': pump1_low,
            'pump1_high_cutoff': pump1_high,
            'turkeys_initial': turkeys_initial,
            'surhs_initial': surhs_initial
        }

        with st.spinner('Running simulation...'):
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            results = system.simulate_reservoir_system(start_date, end_date, params)

        turkeys_min = reservoirs[0].get('min_capacity_ML', 0)
        surhs_min = reservoirs[1].get('min_capacity_ML', 0)

        # Statistics
        total_deficit = results['demand_deficit_ML'].sum()
        deficit_days = (results['demand_deficit_ML'] > 0).sum()
        avg_turkeys = results['turkeys_nest_level_ML'].mean()
        avg_surhs = results['surhs_creek_level_ML'].mean()
        min_surhs = results['surhs_creek_level_ML'].min()
        days_at_min_turkeys = (results['turkeys_nest_level_ML'] <= turkeys_min + 1).sum()
        days_at_min_surhs = (results['surhs_creek_level_ML'] <= surhs_min + 1).sum()

        scenario_desc = "TYPICAL"
        if drought_years:
            scenario_desc += f" + {len(drought_years)} DROUGHT"
        if rain_years:
            scenario_desc += f" + {len(rain_years)} RAIN"

        st.success(f"✓ Simulation Complete: {len(results)} days | Scenario: {scenario_desc}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Demand Deficit", f"{total_deficit:.1f} ML", f"{deficit_days} days")
        col2.metric("Avg Turkeys Nest", f"{avg_turkeys:.1f} ML", f"{avg_turkeys / 1300 * 100:.1f}%")
        col3.metric("Avg Surhs Creek", f"{avg_surhs:.1f} ML", f"{avg_surhs / 1380 * 100:.1f}%")
        col4.metric("Min Surhs Creek", f"{min_surhs:.1f} ML")
        col5.metric("Days at Min", f"TN:{days_at_min_turkeys} SC:{days_at_min_surhs}")

        # [Rest of plotting code remains the same as original...]
        surhs_creek = system.system_config['system']['reservoirs'][1]
        turkeys_nest = system.system_config['system']['reservoirs'][0]

        fig = make_subplots(
            rows=8, cols=1,
            subplot_titles=(
                'Surhs Creek Reservoir Level (ML)',
                'Turkeys Nest Reservoir Level (ML)',
                'River Flow (ML/day)',
                'Evaporation from Reservoirs (ML/day)',
                'Temperature - Ravenswood (°C)',
                'Relative Humidity - Ravenswood (%)',
                'Precipitation - Ravenswood (mm/day)',
                'Inflow to Surhs Creek (ML/day)'
            ),
            vertical_spacing=0.035,
            row_heights=[1.3, 1.3, 1, 1, 1, 1, 1, 1],
            specs=[[{"secondary_y": False}]] * 7 + [[{"secondary_y": True}]]
        )

        # Plots (same as original)
        fig.add_trace(
            go.Scatter(x=results.index, y=results['surhs_creek_level_ML'],
                       name='Surhs Creek', line=dict(color='teal', width=2),
                       fill='tozeroy', fillcolor='rgba(0,128,128,0.2)'),
            row=1, col=1
        )
        fig.add_hline(y=surhs_creek['capacity_ML'], line_dash="dot", line_color="red",
                      annotation_text=f"SC Capacity: {surhs_creek['capacity_ML']} ML", row=1, col=1)
        if surhs_min > 0:
            fig.add_hline(y=surhs_min, line_dash="dash", line_color="orange",
                          annotation_text=f"SC Min: {surhs_min} ML", row=1, col=1)

        fig.add_trace(
            go.Scatter(x=results.index, y=results['turkeys_nest_level_ML'],
                       name='Turkeys Nest', line=dict(color='darkblue', width=2),
                       fill='tozeroy', fillcolor='rgba(0,0,139,0.2)'),
            row=2, col=1
        )
        fig.add_hline(y=turkeys_nest['capacity_ML'], line_dash="dot", line_color="red",
                      annotation_text=f"TN Capacity: {turkeys_nest['capacity_ML']} ML", row=2, col=1)
        if turkeys_min > 0:
            fig.add_hline(y=turkeys_min, line_dash="dash", line_color="orange",
                          annotation_text=f"TN Min: {turkeys_min} ML", row=2, col=1)

        pump_allowed = (results['river_flow_ML_day'] >= pump1_low) & (results['river_flow_ML_day'] <= pump1_high)
        pump_status_display = pump_allowed.astype(int) * 1000000

        fig.add_trace(
            go.Bar(x=results.index, y=pump_status_display,
                   name='Pumping Allowed', marker_color='green', opacity=0.8),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=results.index, y=results['river_flow_ML_day'],
                       name='River Flow', line=dict(color='darkblue', width=1.5, shape='hv')),
            row=3, col=1
        )
        fig.add_hline(y=pump1_low, line_dash="dash", line_color="grey", row=3, col=1)
        fig.add_hline(y=pump1_high, line_dash="dash", line_color="grey", row=3, col=1)

        fig.add_trace(
            go.Scatter(x=results.index, y=results['turkeys_evap_ML_day'],
                       name='Evap - Turkeys Nest', line=dict(color='orange', width=1.5)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=results.index, y=results['surhs_evap_ML_day'],
                       name='Evap - Surhs Creek', line=dict(color='darkorange', width=1.5)),
            row=4, col=1
        )

        fig.add_trace(
            go.Scatter(x=results.index, y=results['temperature_degC'],
                       name='Temperature', line=dict(color='orangered', width=1.5)),
            row=5, col=1
        )

        fig.add_trace(
            go.Scatter(x=results.index, y=results['relative_humidity_pct'],
                       name='Relative Humidity', line=dict(color='blue', width=1.5)),
            row=6, col=1
        )

        precip_normal = results['precipitation_mm_day'].copy()
        precip_normal[precip_normal > 30] = np.nan

        fig.add_trace(
            go.Bar(x=results.index, y=precip_normal,
                   name='Precipitation', marker_color='navy', opacity=0.6),
            row=7, col=1
        )

        precip_ma = results['precipitation_mm_day'].rolling(window=120, center=True).mean()
        fig.add_trace(
            go.Scatter(x=results.index, y=precip_ma,
                       name='120-day MA Precipitation',
                       line=dict(color='blue', width=2, dash='dash')),
            row=7, col=1
        )

        exceed_dates = results.index[results['precipitation_mm_day'] > 30]
        exceed_values = results['precipitation_mm_day'][results['precipitation_mm_day'] > 30]
        if len(exceed_dates) > 0:
            fig.add_trace(
                go.Scatter(x=exceed_dates, y=[30] * len(exceed_dates),
                           mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'),
                           name='Exceeds 30mm',
                           text=[f'{val:.1f}mm' for val in exceed_values]),
                row=7, col=1
            )

        fig.add_hline(y=surhs_creek['inflows']['fluvial']['min_rain_mm'],
                      line_dash="dash", line_color="orange", row=7, col=1)

        fig.add_trace(
            go.Scatter(x=results.index, y=results['fluvial_inflow_ML'],
                       name='Fluvial Inflow', line=dict(color='darkgreen', width=1.5),
                       stackgroup='inflow'),
            row=8, col=1
        )
        fig.add_trace(
            go.Scatter(x=results.index, y=results['pluvial_inflow_surhs_ML'],
                       name='Pluvial Inflow (Surhs)', line=dict(color='lightgreen', width=1.5),
                       stackgroup='inflow'),
            row=8, col=1
        )

        total_inflow_ma = results['total_inflow_ML'].rolling(window=120, center=True).mean()
        fig.add_trace(
            go.Scatter(x=results.index, y=total_inflow_ma,
                       name='120-day MA Total Inflow',
                       line=dict(color='darkgreen', width=2, dash='dash')),
            row=8, col=1
        )

        # Update axes
        fig.update_xaxes(title_text="Date", row=8, col=1)
        fig.update_yaxes(title_text="ML", range=[0, surhs_creek['capacity_ML'] * 1.1], row=1, col=1)
        fig.update_yaxes(title_text="ML", range=[0, turkeys_nest['capacity_ML'] * 1.1], row=2, col=1)
        fig.update_yaxes(title_text="ML/day", type="log", range=[2, 6], row=3, col=1)
        fig.update_yaxes(title_text="ML/day", row=4, col=1)
        fig.update_yaxes(title_text="°C", row=5, col=1)
        fig.update_yaxes(title_text="%", row=6, col=1)
        fig.update_yaxes(title_text="mm", range=[0, 30], row=7, col=1)
        fig.update_yaxes(title_text="ML/day", range=[0, 5], row=8, col=1)

        fig.update_layout(
            height=4200,
            showlegend=True,
            title_text=f"Reservoir System Analysis - {scenario_desc} ({start_year}-{end_year})",
            hovermode='x unified',
            template='plotly_white',
            bargap=0,
            bargroupgap=0
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download results
        with st.expander("📥 Download Results"):
            csv = results.to_csv()
            st.download_button(
                label="Download simulation results (CSV)",
                data=csv,
                file_name=f"reservoir_sim_{start_year}_{end_year}_{scenario_desc.replace(' ', '_')}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)

else:
    st.info("👈 Configure simulation parameters and click 'Run Simulation'")

    # Show available scenarios
    with st.expander("ℹ️ About Flow Scenarios"):
        st.markdown("""
        ### Flow Scenario System

        **TYPICAL (default):**
        - Sampled from middle 80% of hydrological years by total flow
        - Represents "business as usual" conditions
        - Excludes extreme drought and flood years

        **DROUGHT insertion:**
        - Replaces specified hydrological year with driest year from bottom 10%
        - Hydrological year = Nov (year-1) to Oct (year)
        - Example: Drought 2050 = Nov 2049 to Oct 2050

        **EXTREME RAIN insertion:**
        - Replaces specified hydrological year with wettest year from top 10%
        - Tests reservoir capacity under flood conditions

        **Usage:**
        1. Base simulation uses TYPICAL scenario
        2. Add drought/rain years to test extreme conditions
        3. Multiple years can be inserted (comma-separated)
        4. Each insertion replaces one complete hydrological year
        """)

with st.expander("💡 Tips"):
    st.markdown("""
    - **Baseline test:** Run with TYPICAL scenario to establish normal performance
    - **Drought stress test:** Insert drought years to test water security
    - **Flood stress test:** Insert rain years to test overflow/capacity
    - **Combined scenarios:** Test multiple extreme years (e.g., "2030, 2040, 2050")
    - **Hydrological years:** Remember Nov-Oct span, not Jan-Dec
    - **Minimum capacity:** Dead storage protects pump intake and emergency reserves
    - **Compare scenarios:** Run multiple times with different extreme years to compare
    """)