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
        r1_turkeys = reservoirs[0]
        r2_surhs = reservoirs[1]

        r1_min_capacity = r1_turkeys.get('min_capacity_ML', 0)
        r2_min_capacity = r2_surhs.get('min_capacity_ML', 0)

        # Initialize arrays
        n = len(data)
        r1_level = np.zeros(n)
        r2_level = np.zeros(n)
        pump_river_to_r1 = np.zeros(n)
        pump_r1_to_r2 = np.zeros(n)
        pump_r2_to_site = np.zeros(n)
        fluvial_inflow = np.zeros(n)
        pluvial_inflow_r1 = np.zeros(n)
        pluvial_inflow_r2 = np.zeros(n)
        total_inflow = np.zeros(n)
        r1_evap_loss = np.zeros(n)
        r2_evap_loss = np.zeros(n)
        demand_supplied = np.zeros(n)
        demand_deficit = np.zeros(n)

        # Initial levels
        r1_level[0] = params.get('r1_initial', r1_turkeys['initial_level_ML'])
        r2_level[0] = params.get('r2_initial', r2_surhs['initial_level_ML'])

        # PUMP RATE PARAMETERS - CLEARLY NAMED
        pump_river_to_r1_max = params.get('pump_river_to_r1_max', pumps[0]['max_rate_in_ML_day'])
        pump_r1_to_r2_max = params.get('pump_r1_to_r2_max', pumps[0]['max_rate_out_ML_day'])
        pump_r2_to_site_max = params.get('pump_r2_to_site_max', pumps[1]['max_rate_out_ML_day'])

        # RIVER FLOW CUTOFFS FOR PUMP_RIVER_TO_R1
        low_cutoff = params.get('river_pump_low_cutoff', pumps[0]['cutoffs']['low_flow_ML_day'])
        high_cutoff = params.get('river_pump_high_cutoff', pumps[0]['cutoffs']['high_flow_ML_day'])

        demand_ML_day = params.get('demand_ML_day', 9.8)

        # Simulation loop
        for i in range(1, n):
            r1_current = r1_level[i - 1]
            r2_current = r2_level[i - 1]

            # =====================================================
            # PUMP: RIVER → R1 (Turkeys Nest)
            # =====================================================
            available_capacity_r1 = r1_turkeys['capacity_ML'] - r1_current
            if river_flow[i] >= low_cutoff and river_flow[i] <= high_cutoff and available_capacity_r1 > 0:
                pump_river_to_r1[i] = min(pump_river_to_r1_max, available_capacity_r1)
            else:
                pump_river_to_r1[i] = 0

            # =====================================================
            # FLUVIAL INFLOW (to R2 only)
            # =====================================================
            if precipitation[i] >= r2_surhs['inflows']['fluvial']['min_rain_mm']:
                fluvial_inflow[i] = precipitation[i] * r2_surhs['inflows']['fluvial']['coefficient_ML_per_mm']
            else:
                fluvial_inflow[i] = 0

            # =====================================================
            # PLUVIAL INFLOW (direct rainfall on reservoir surfaces)
            # =====================================================
            if precipitation[i] >= 2.0:
                pluvial_inflow_r1[i] = (precipitation[i] / 1000) * r1_turkeys['surface_area_m²'] / 1000
                pluvial_inflow_r2[i] = (precipitation[i] / 1000) * r2_surhs['surface_area_m²'] / 1000
            else:
                pluvial_inflow_r1[i] = 0
                pluvial_inflow_r2[i] = 0

            total_inflow[i] = fluvial_inflow[i] + pluvial_inflow_r2[i]

            # =====================================================
            # EVAPORATION LOSSES
            # =====================================================
            r1_evap_loss[i] = (evaporation_mm[i] / 1000) * r1_turkeys['surface_area_m²'] / 1000
            r2_evap_loss[i] = (evaporation_mm[i] / 1000) * r2_surhs['surface_area_m²'] / 1000

            # =====================================================
            # UPDATE R1 (TURKEYS NEST)
            # =====================================================
            r1_current += pump_river_to_r1[i] + pluvial_inflow_r1[i]
            r1_current -= r1_turkeys['losses']['seepage_ML_day'] + r1_evap_loss[i]

            # =====================================================
            # PUMP: R1 → R2 (Turkeys Nest → Surhs Creek)
            # =====================================================
            available_water_r1 = max(0, r1_current - r1_min_capacity)
            pump_r1_to_r2[i] = min(pump_r1_to_r2_max, available_water_r1)
            pump_r1_to_r2[i] = max(0, pump_r1_to_r2[i])

            r1_current -= pump_r1_to_r2[i]
            r1_current = max(r1_min_capacity, min(r1_current, r1_turkeys['capacity_ML']))
            r1_level[i] = r1_current

            # =====================================================
            # UPDATE R2 (SURHS CREEK)
            # =====================================================
            r2_current += pump_r1_to_r2[i] + fluvial_inflow[i] + pluvial_inflow_r2[i]
            r2_current -= r2_surhs['losses']['seepage_ML_day'] + r2_evap_loss[i]

            # =====================================================
            # PUMP: R2 → SITE (Surhs Creek → Demand)
            # =====================================================
            available_water_r2 = max(0, r2_current - r2_min_capacity)
            pump_r2_to_site[i] = min(pump_r2_to_site_max, demand_ML_day, available_water_r2)
            pump_r2_to_site[i] = max(0, pump_r2_to_site[i])

            r2_current -= pump_r2_to_site[i]

            demand_supplied[i] = pump_r2_to_site[i]
            demand_deficit[i] = demand_ML_day - pump_r2_to_site[i]

            r2_current = max(r2_min_capacity, min(r2_current, r2_surhs['capacity_ML']))
            r2_level[i] = r2_current

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
            'pluvial_inflow_r1_ML': pluvial_inflow_r1,
            'pluvial_inflow_r2_ML': pluvial_inflow_r2,
            'total_inflow_ML': total_inflow,
            'r1_evap_ML_day': r1_evap_loss,
            'r2_evap_ML_day': r2_evap_loss,
            'pump_river_to_r1_ML_day': pump_river_to_r1,
            'pump_r1_to_r2_ML_day': pump_r1_to_r2,
            'pump_r2_to_site_ML_day': pump_r2_to_site,
            'r1_level_ML': r1_level,
            'r2_level_ML': r2_level,
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
    st.markdown("**Pump: River → R1 (Turkeys Nest)**")
    pump_river_to_r1_max = st.number_input(
        "Max Rate (ML/day)",
        min_value=0,
        value=int(pumps[0]['max_rate_in_ML_day']),
        step=1,
        key="pump_river_r1"
    )

    st.markdown("**Pump: R1 → R2 (Turkeys → Surhs)**")
    pump_r1_to_r2_max = st.number_input(
        "Max Rate (ML/day)",
        min_value=0.0,
        value=float(pumps[0]['max_rate_out_ML_day']),
        step=0.1,
        key="pump_r1_r2"
    )

    st.markdown("**Pump: R2 → Site (Surhs → Demand)**")
    pump_r2_to_site_max = st.number_input(
        "Max Rate (ML/day)",
        min_value=0.0,
        value=float(pumps[1]['max_rate_out_ML_day']),
        step=0.1,
        key="pump_r2_site",
        help="Physical pump capacity limit"
    )

with st.sidebar.expander("🎯 River Flow Cutoffs (Pump River→R1)", expanded=False):
    river_pump_low = st.number_input(
        "Low Cutoff (ML/day)",
        min_value=0,
        value=int(pumps[0]['cutoffs']['low_flow_ML_day']),
        step=1,
        help="Minimum river flow to allow pumping"
    )
    river_pump_high = st.number_input(
        "High Cutoff (ML/day)",
        min_value=0,
        value=int(pumps[0]['cutoffs']['high_flow_ML_day']),
        step=100,
        help="Maximum river flow to allow pumping"
    )

with st.sidebar.expander("🏞️ Initial Reservoir Levels", expanded=False):
    r1_initial = st.number_input(
        f"R1 - Turkeys Nest (ML, max {reservoirs[0]['capacity_ML']})",
        min_value=0, max_value=int(reservoirs[0]['capacity_ML']),
        value=int(reservoirs[0]['initial_level_ML']), step=10
    )
    r2_initial = st.number_input(
        f"R2 - Surhs Creek (ML, max {reservoirs[1]['capacity_ML']})",
        min_value=0, max_value=int(reservoirs[1]['capacity_ML']),
        value=int(reservoirs[1]['initial_level_ML']), step=10
    )

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
            'pump_river_to_r1_max': pump_river_to_r1_max,
            'pump_r1_to_r2_max': pump_r1_to_r2_max,
            'pump_r2_to_site_max': pump_r2_to_site_max,
            'river_pump_low_cutoff': river_pump_low,
            'river_pump_high_cutoff': river_pump_high,
            'r1_initial': r1_initial,
            'r2_initial': r2_initial
        }

        with st.spinner('Running simulation...'):
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            results = system.simulate_reservoir_system(start_date, end_date, params)

        r1_min = reservoirs[0].get('min_capacity_ML', 0)
        r2_min = reservoirs[1].get('min_capacity_ML', 0)

        # Statistics
        total_deficit = results['demand_deficit_ML'].sum()
        deficit_days = (results['demand_deficit_ML'] > 0).sum()
        avg_r1 = results['r1_level_ML'].mean()
        avg_r2 = results['r2_level_ML'].mean()
        min_r2 = results['r2_level_ML'].min()
        days_at_min_r1 = (results['r1_level_ML'] <= r1_min + 1).sum()
        days_at_min_r2 = (results['r2_level_ML'] <= r2_min + 1).sum()

        # Pump diagnostics
        days_pump_r1_r2_zero = (results['pump_r1_to_r2_ML_day'] == 0).sum()
        days_pump_river_r1_active = (results['pump_river_to_r1_ML_day'] > 0).sum()
        avg_pump_r1_r2 = results['pump_r1_to_r2_ML_day'].mean()
        total_fluvial = results['fluvial_inflow_ML'].sum()
        total_pluvial_r2 = results['pluvial_inflow_r2_ML'].sum()

        scenario_desc = "TYPICAL"
        if drought_years:
            scenario_desc += f" + {len(drought_years)} DROUGHT"
        if rain_years:
            scenario_desc += f" + {len(rain_years)} RAIN"

        st.success(f"✓ Simulation Complete: {len(results)} days | Scenario: {scenario_desc}")

        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Demand Deficit", f"{total_deficit:.1f} ML", f"{deficit_days} days")
        col2.metric("Avg R1 (Turkeys)", f"{avg_r1:.1f} ML", f"{avg_r1 / 1300 * 100:.1f}%")
        col3.metric("Avg R2 (Surhs)", f"{avg_r2:.1f} ML", f"{avg_r2 / 1380 * 100:.1f}%")
        col4.metric("Min R2 (Surhs)", f"{min_r2:.1f} ML")
        col5.metric("Days at Min", f"R1:{days_at_min_r1} R2:{days_at_min_r2}")

        # Pump diagnostics
        st.markdown("### 🔍 Pump Activity Diagnostics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Days R1→R2=0", f"{days_pump_r1_r2_zero}",
                    f"{days_pump_r1_r2_zero / len(results) * 100:.1f}% of time")
        col2.metric("Days River→R1 Active", f"{days_pump_river_r1_active}",
                    f"{days_pump_river_r1_active / len(results) * 100:.1f}% of time")
        col3.metric("Avg R1→R2 Flow", f"{avg_pump_r1_r2:.2f} ML/day")
        col4.metric("Total Fluvial (R2)", f"{total_fluvial:.0f} ML")
        col5.metric("Total Pluvial (R2)", f"{total_pluvial_r2:.0f} ML")

        # [PLOTTING CODE - Updated with new names]
        r1_turkeys = system.system_config['system']['reservoirs'][0]
        r2_surhs = system.system_config['system']['reservoirs'][1]

        fig = make_subplots(
            rows=8, cols=1,
            subplot_titles=(
                'R2 (Surhs Creek) Reservoir Level (ML)',
                'R1 (Turkeys Nest) Reservoir Level (ML)',
                'River Flow (ML/day)',
                'Evaporation from Reservoirs (ML/day)',
                'Temperature - Ravenswood (°C)',
                'Relative Humidity - Ravenswood (%)',
                'Precipitation - Ravenswood (mm/day)',
                'Inflow to R2 - Surhs Creek (ML/day)'
            ),
            vertical_spacing=0.035,
            row_heights=[1.3, 1.3, 1, 1, 1, 1, 1, 1],
            specs=[[{"secondary_y": False}]] * 7 + [[{"secondary_y": True}]]
        )

        # R2 (Surhs Creek) Level
        fig.add_trace(
            go.Scatter(x=results.index, y=results['r2_level_ML'],
                       name='R2 (Surhs)', line=dict(color='teal', width=2),
                       fill='tozeroy', fillcolor='rgba(0,128,128,0.2)'),
            row=1, col=1
        )
        fig.add_hline(y=r2_surhs['capacity_ML'], line_dash="dot", line_color="red",
                      annotation_text=f"R2 Capacity: {r2_surhs['capacity_ML']} ML", row=1, col=1)
        if r2_min > 0:
            fig.add_hline(y=r2_min, line_dash="dash", line_color="orange",
                          annotation_text=f"R2 Min: {r2_min} ML", row=1, col=1)

        # R1 (Turkeys Nest) Level
        fig.add_trace(
            go.Scatter(x=results.index, y=results['r1_level_ML'],
                       name='R1 (Turkeys)', line=dict(color='darkblue', width=2),
                       fill='tozeroy', fillcolor='rgba(0,0,139,0.2)'),
            row=2, col=1
        )
        fig.add_hline(y=r1_turkeys['capacity_ML'], line_dash="dot", line_color="red",
                      annotation_text=f"R1 Capacity: {r1_turkeys['capacity_ML']} ML", row=2, col=1)
        if r1_min > 0:
            fig.add_hline(y=r1_min, line_dash="dash", line_color="orange",
                          annotation_text=f"R1 Min: {r1_min} ML", row=2, col=1)

        # River Flow with Pumping Window
        pump_allowed = (results['river_flow_ML_day'] >= river_pump_low) & (
                    results['river_flow_ML_day'] <= river_pump_high)
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
        fig.add_hline(y=river_pump_low, line_dash="dash", line_color="grey", row=3, col=1)
        fig.add_hline(y=river_pump_high, line_dash="dash", line_color="grey", row=3, col=1)

        # Evaporation
        fig.add_trace(
            go.Scatter(x=results.index, y=results['r1_evap_ML_day'],
                       name='Evap - R1 (Turkeys)', line=dict(color='orange', width=1.5)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=results.index, y=results['r2_evap_ML_day'],
                       name='Evap - R2 (Surhs)', line=dict(color='darkorange', width=1.5)),
            row=4, col=1
        )

        # Temperature
        fig.add_trace(
            go.Scatter(x=results.index, y=results['temperature_degC'],
                       name='Temperature', line=dict(color='orangered', width=1.5)),
            row=5, col=1
        )

        # Humidity
        fig.add_trace(
            go.Scatter(x=results.index, y=results['relative_humidity_pct'],
                       name='Relative Humidity', line=dict(color='blue', width=1.5)),
            row=6, col=1
        )

        # Precipitation
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

        fig.add_hline(y=r2_surhs['inflows']['fluvial']['min_rain_mm'],
                      line_dash="dash", line_color="orange", row=7, col=1)

        # Inflow to R2
        fig.add_trace(
            go.Scatter(x=results.index, y=results['fluvial_inflow_ML'],
                       name='Fluvial Inflow', line=dict(color='darkgreen', width=1.5),
                       stackgroup='inflow'),
            row=8, col=1
        )
        fig.add_trace(
            go.Scatter(x=results.index, y=results['pluvial_inflow_r2_ML'],
                       name='Pluvial Inflow (R2)', line=dict(color='lightgreen', width=1.5),
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
        fig.update_yaxes(title_text="ML", range=[0, r2_surhs['capacity_ML'] * 1.1], row=1, col=1)
        fig.update_yaxes(title_text="ML", range=[0, r1_turkeys['capacity_ML'] * 1.1], row=2, col=1)
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

    # Show system diagram
    with st.expander("🔧 System Architecture"):
        st.markdown("""
        ### Reservoir System Flow

        ```
        RIVER (Burdekin)
           ↓ 
        [Pump: River→R1] (Max: 34 ML/day, operates when 543 < flow < 100,000 ML/day)
           ↓
        R1 (Turkeys Nest) - Capacity: 1,300 ML, Min: 150 ML
           ↓
        [Pump: R1→R2] (Max: 10.5 ML/day)
           ↓
        R2 (Surhs Creek) - Capacity: 1,380 ML, Min: 180 ML
           ↓  (+ Fluvial/Pluvial inflows)
        [Pump: R2→Site] (Max: 10.0 ML/day)
           ↓
        DEMAND SITE (9.8 ML/day baseline)
        ```

        **Key Constraints:**
        - River pump only operates within specific flow window
        - R1 must stay above 150 ML (dead storage)
        - R2 must stay above 180 ML (dead storage)
        - Final delivery pump limited to 10.0 ML/day
        """)

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
        """)

with st.expander("💡 Tips"):
    st.markdown("""
    - **Clear naming:** River→R1→R2→Site represents the complete water path
    - **Test pump constraints:** Increase demand above 10 ML/day to see pump R2→Site limitation
    - **Watch R1→R2 diagnostics:** "Days R1→R2=0" shows when Turkeys Nest is too empty to supply
    - **Drought scenarios:** Insert drought years to see extended periods of R1 depletion
    - **Starting conditions:** Try lower initial levels (e.g., 50%) to test system resilience
    """)