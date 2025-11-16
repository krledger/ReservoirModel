"""
Core reservoir operations and simulation logic
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path


class ReservoirSystem:
    """Main class to handle reservoir system simulation"""

    def __init__(self, base_path='.'):
        self.base_path = Path(base_path)
        self.climate_data = None
        self.flow_data = None
        self.system_config = None
    
    @staticmethod
    def get_surface_area(reservoir_config):
        """Safely get surface area from config with different possible key names"""
        # First, try to find any key that contains 'surface_area'
        for key in reservoir_config.keys():
            if 'surface_area' in key.lower():
                return reservoir_config[key]
        
        # Fallback to checking specific variations
        possible_keys = [
            'surface_area_mÂ²',
            'surface_area_m2',
            'surface_area_mÃ‚Â²',
            'surface_area_mÃƒâ€šÃ‚Â²',  # UTF-8 encoding issue
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

    def generate_smoothed_random_multipliers(self, n_days, mean=1.0, variation=0.1, alpha=0.85, seed=None):
        """
        Generate smoothed random multipliers using AR(1) process.

        Parameters:
        - n_days: number of days to generate
        - mean: centre point (1.0 = no change)
        - variation: maximum deviation from mean (e.g., 0.1 = Ãƒâ€šÃ‚Â±10%)
        - alpha: autocorrelation coefficient (0.85 = weekly correlation)
        - seed: random seed for reproducibility

        Returns:
        - array of smoothed multipliers
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate random innovations
        innovations = np.random.normal(0, variation / 3, n_days)

        # Apply AR(1) process for smooth transitions
        multipliers = np.zeros(n_days)
        multipliers[0] = mean + innovations[0]

        for i in range(1, n_days):
            multipliers[i] = alpha * multipliers[i - 1] + (1 - alpha) * mean + innovations[i]

        # Clip to ensure we stay within bounds
        lower_bound = mean - variation
        upper_bound = mean + variation
        multipliers = np.clip(multipliers, lower_bound, upper_bound)

        return multipliers

    def load_system_config(self, config_path='reservoir_system.json'):
        """Load reservoir system configuration"""
        config_file = self.base_path / config_path
        if not config_file.exists():
            raise FileNotFoundError(f"System configuration not found: {config_file}")

        with open(config_file, 'r') as f:
            self.system_config = json.load(f)
        return self.system_config

    def load_climate_data(self, scenario='SSP1-26'):
        """Load climate data from parquet file based on scenario"""
        climate_base = self.base_path / 'metricsDataFiles'

        if not climate_base.exists():
            raise FileNotFoundError(f"Climate data directory not found: {climate_base}")

        parquet_file = None

        # Look for scenario-specific file first
        scenario_patterns = [
            f'raw_daily_{scenario}.parquet',
            f'{scenario}_raw_daily.parquet',
            f'raw_daily.parquet'
        ]

        for root, dirs, files in os.walk(climate_base):
            for pattern in scenario_patterns:
                if pattern in files:
                    parquet_file = Path(root) / pattern
                    break
            if parquet_file:
                break

        if not parquet_file or not parquet_file.exists():
            raise FileNotFoundError(f"Climate data file not found for scenario '{scenario}' in {climate_base}")

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

        # Initialise arrays
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

        # Pump rate parameters
        pump_river_to_r1_max = params.get('pump_river_to_r1_max', pumps[0]['max_rate_in_ML_day'])
        pump_r1_to_r2_max = params.get('pump_r1_to_r2_max', pumps[0]['max_rate_out_ML_day'])
        pump_r2_to_site_max = params.get('pump_r2_to_site_max', pumps[1]['max_rate_out_ML_day'])

        # River flow cutoffs
        low_cutoff = params.get('river_pump_low_cutoff', pumps[0]['cutoffs']['low_flow_ML_day'])
        high_cutoff = params.get('river_pump_high_cutoff', pumps[0]['cutoffs']['high_flow_ML_day'])

        demand_ML_day = params.get('demand_ML_day', 9.8)

        # Random variations
        enable_random = params.get('enable_random', False)
        if enable_random:
            random_seed = params.get('random_seed', 42)
            demand_multipliers = self.generate_smoothed_random_multipliers(n, mean=1.0, variation=0.2, alpha=0.85,
                                                                           seed=random_seed)
            pump_river_r1_multipliers = self.generate_smoothed_random_multipliers(n, mean=1.0, variation=0.1,
                                                                                  alpha=0.85, seed=random_seed + 1)
            pump_r1_r2_multipliers = self.generate_smoothed_random_multipliers(n, mean=1.0, variation=0.1, alpha=0.85,
                                                                               seed=random_seed + 2)
            pump_r2_site_multipliers = self.generate_smoothed_random_multipliers(n, mean=1.0, variation=0.1, alpha=0.85,
                                                                                 seed=random_seed + 3)
        else:
            demand_multipliers = np.ones(n)
            pump_river_r1_multipliers = np.ones(n)
            pump_r1_r2_multipliers = np.ones(n)
            pump_r2_site_multipliers = np.ones(n)

        # Simulation loop
        for i in range(1, n):
            r1_current = r1_level[i - 1]
            r2_current = r2_level[i - 1]

            # Pump: River ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ R1
            available_capacity_r1 = r1_turkeys['capacity_ML'] - r1_current
            if available_capacity_r1 <= 0:
                pump_river_to_r1[i] = 0
            elif river_flow[i] >= low_cutoff and river_flow[i] <= high_cutoff:
                actual_pump_max = pump_river_to_r1_max * pump_river_r1_multipliers[i]
                pump_river_to_r1[i] = min(actual_pump_max, available_capacity_r1)
            else:
                pump_river_to_r1[i] = 0

            # Fluvial inflow
            if precipitation[i] >= r2_surhs['inflows']['fluvial']['min_rain_mm']:
                fluvial_inflow[i] = precipitation[i] * r2_surhs['inflows']['fluvial']['coefficient_ML_per_mm']
            else:
                fluvial_inflow[i] = 0

            # Pluvial inflow
            if precipitation[i] >= 2.0:
                pluvial_inflow_r1[i] = (precipitation[i] / 1000) * self.get_surface_area(r1_turkeys) / 1000
                pluvial_inflow_r2[i] = (precipitation[i] / 1000) * self.get_surface_area(r2_surhs) / 1000
            else:
                pluvial_inflow_r1[i] = 0
                pluvial_inflow_r2[i] = 0

            total_inflow[i] = fluvial_inflow[i] + pluvial_inflow_r2[i]

            # Evaporation losses
            r1_evap_loss[i] = (evaporation_mm[i] / 1000) * self.get_surface_area(r1_turkeys) / 1000
            r2_evap_loss[i] = (evaporation_mm[i] / 1000) * self.get_surface_area(r2_surhs) / 1000

            # Update R1
            r1_current += pump_river_to_r1[i] + pluvial_inflow_r1[i]
            r1_current -= r1_turkeys['losses']['seepage_ML_day'] + r1_evap_loss[i]

            # Pump: R1 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ R2
            available_water_r1 = max(0, r1_current - r1_min_capacity)
            available_capacity_r2 = max(0, r2_surhs['capacity_ML'] - r2_current)

            if available_water_r1 <= 0 or available_capacity_r2 <= 0:
                pump_r1_to_r2[i] = 0
            else:
                actual_pump_r1_r2_max = pump_r1_to_r2_max * pump_r1_r2_multipliers[i]
                r1_target = r1_turkeys['capacity_ML'] * 0.70
                r2_target = r2_surhs['capacity_ML'] * 0.70
                r1_high = r1_turkeys['capacity_ML'] * 0.85
                r2_low = r2_surhs['capacity_ML'] * 0.50

                r2_deficit = max(0, r2_target - r2_current)
                r1_excess = max(0, r1_current - r1_target)

                if r2_current < r2_low:
                    pump_for_r2 = min(actual_pump_r1_r2_max, r2_deficit, available_water_r1, available_capacity_r2)
                else:
                    pump_for_r2 = 0

                if r1_current > r1_high:
                    remaining_pump_capacity = actual_pump_r1_r2_max - pump_for_r2
                    pump_for_r1 = min(remaining_pump_capacity, r1_excess, available_water_r1, available_capacity_r2)
                else:
                    pump_for_r1 = 0

                pump_r1_to_r2[i] = pump_for_r2 + pump_for_r1
                pump_r1_to_r2[i] = max(0, min(pump_r1_to_r2[i], actual_pump_r1_r2_max))
                pump_r1_to_r2[i] = min(pump_r1_to_r2[i], available_capacity_r2)

            r1_current -= pump_r1_to_r2[i]
            r1_current = max(r1_min_capacity, min(r1_current, r1_turkeys['capacity_ML']))
            r1_level[i] = r1_current

            # Update R2
            r2_current += pump_r1_to_r2[i] + fluvial_inflow[i] + pluvial_inflow_r2[i]
            r2_current -= r2_surhs['losses']['seepage_ML_day'] + r2_evap_loss[i]

            # Pump: R2 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Site
            available_water_r2 = max(0, r2_current - r2_min_capacity)
            actual_pump_r2_site_max = pump_r2_to_site_max * pump_r2_site_multipliers[i]
            actual_demand = demand_ML_day * demand_multipliers[i]

            if available_water_r2 <= 0:
                pump_r2_to_site[i] = 0
            else:
                pump_r2_to_site[i] = min(actual_pump_r2_site_max, actual_demand, available_water_r2)

            pump_r2_to_site[i] = max(0, pump_r2_to_site[i])
            r2_current -= pump_r2_to_site[i]

            demand_supplied[i] = pump_r2_to_site[i]
            demand_deficit[i] = actual_demand - pump_r2_to_site[i]

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


def calculate_statistics(results, reservoirs, pumps=None):
    """Calculate standard statistics from simulation results
    
    Args:
        results: DataFrame with simulation results
        reservoirs: List of reservoir configs
        pumps: Optional list of pump configs for utilization calculations
    """
    r1_min = reservoirs[0].get('min_capacity_ML', 0)
    r2_min = reservoirs[1].get('min_capacity_ML', 0)
    
    # Calculate pump utilization (annual) if pump config provided
    total_days = len(results)
    days_per_year = 365.25
    
    stats = {
        'total_deficit': results['demand_deficit_ML'].sum(),
        'deficit_days': (results['demand_deficit_ML'] > 0).sum(),
        'avg_r1': results['r1_level_ML'].mean(),
        'avg_r2': results['r2_level_ML'].mean(),
        'min_r2': results['r2_level_ML'].min(),
        'days_at_min_r1': (results['r1_level_ML'] <= r1_min + 1).sum(),
        'days_at_min_r2': (results['r2_level_ML'] <= r2_min + 1).sum(),
        'days_pump_r1_r2_zero': (results['pump_r1_to_r2_ML_day'] == 0).sum() if 'pump_r1_to_r2_ML_day' in results.columns else 0,
        'days_pump_river_r1_active': (results['pump_river_to_r1_ML_day'] > 0).sum() if 'pump_river_to_r1_ML_day' in results.columns else 0,
        'avg_pump_r1_r2': results['pump_r1_to_r2_ML_day'].mean() if 'pump_r1_to_r2_ML_day' in results.columns else 0,
        'avg_pump_river_r1': results['pump_river_to_r1_ML_day'].mean() if 'pump_river_to_r1_ML_day' in results.columns else 0,
        'avg_pump_r2_site': results['pump_r2_to_site_ML_day'].mean() if 'pump_r2_to_site_ML_day' in results.columns else 0,
        'total_fluvial': results['fluvial_inflow_ML'].sum() if 'fluvial_inflow_ML' in results.columns else 0,
        'total_pluvial_r2': results['pluvial_inflow_r2_ML'].sum() if 'pluvial_inflow_r2_ML' in results.columns else 0,
    }
    
    # Calculate days of storage for each reservoir and combined
    # Get seepage and evaporation rates
    r1_seepage = reservoirs[0]['losses']['seepage_ML_day']
    r2_seepage = reservoirs[1]['losses']['seepage_ML_day']
    avg_r1_evap = results['r1_evap_ML_day'].mean() if 'r1_evap_ML_day' in results.columns else 0
    avg_r2_evap = results['r2_evap_ML_day'].mean() if 'r2_evap_ML_day' in results.columns else 0
    avg_demand = stats['avg_pump_r2_site']
    
    # R1 (Turkeys Nest) days of storage
    # R1 loses water to: seepage, evaporation, and pumping to R2
    available_r1 = stats['avg_r1'] - r1_min
    r1_daily_loss = r1_seepage + avg_r1_evap
    if r1_daily_loss > 0:
        stats['days_of_storage_r1'] = available_r1 / r1_daily_loss
    else:
        stats['days_of_storage_r1'] = 0
    
    # R2 (Surhs Creek) days of storage
    # R2 loses water to: seepage, evaporation, and demand
    available_r2 = stats['avg_r2'] - r2_min
    r2_daily_loss = r2_seepage + avg_r2_evap + avg_demand
    if r2_daily_loss > 0:
        stats['days_of_storage_r2'] = available_r2 / r2_daily_loss
    else:
        stats['days_of_storage_r2'] = 0
    
    # Combined system days of storage
    # Total available storage across both reservoirs
    # Total consumption = demand + all losses
    available_total = available_r1 + available_r2
    total_daily_consumption = avg_demand + r1_seepage + r2_seepage + avg_r1_evap + avg_r2_evap
    
    if total_daily_consumption > 0:
        stats['days_of_storage_total'] = available_total / total_daily_consumption
    else:
        stats['days_of_storage_total'] = 0
    
    # Add detailed breakdown for diagnostic display
    stats['daily_consumption_breakdown'] = {
        'demand': avg_demand,
        'r1_seepage': r1_seepage,
        'r1_evap': avg_r1_evap,
        'r1_losses': r1_seepage + avg_r1_evap,
        'r2_seepage': r2_seepage,
        'r2_evap': avg_r2_evap,
        'r2_losses': r2_seepage + avg_r2_evap,
        'total_loss': total_daily_consumption,
        'r1_available': available_r1,
        'r2_available': available_r2,
        'total_available': available_total
    }
    
    # Calculate annual utilization if pumps config provided
    if pumps is not None and len(pumps) >= 2:
        # River -> R1 utilization
        if 'pump_river_to_r1_ML_day' in results.columns:
            avg_flow = results['pump_river_to_r1_ML_day'].mean()
            max_capacity = pumps[0]['max_rate_in_ML_day']
            stats['utilization_river_r1_pct'] = (avg_flow / max_capacity * 100) if max_capacity > 0 else 0
            
            # Annual volume
            stats['annual_vol_river_r1_ML'] = avg_flow * days_per_year
        else:
            stats['utilization_river_r1_pct'] = 0
            stats['annual_vol_river_r1_ML'] = 0
        
        # R1 -> R2 utilization
        if 'pump_r1_to_r2_ML_day' in results.columns:
            avg_flow = results['pump_r1_to_r2_ML_day'].mean()
            max_capacity = pumps[0]['max_rate_out_ML_day']
            stats['utilization_r1_r2_pct'] = (avg_flow / max_capacity * 100) if max_capacity > 0 else 0
            
            # Annual volume
            stats['annual_vol_r1_r2_ML'] = avg_flow * days_per_year
        else:
            stats['utilization_r1_r2_pct'] = 0
            stats['annual_vol_r1_r2_ML'] = 0
        
        # R2 -> Site utilization
        if 'pump_r2_to_site_ML_day' in results.columns:
            avg_flow = results['pump_r2_to_site_ML_day'].mean()
            max_capacity = pumps[1]['max_rate_out_ML_day']
            stats['utilization_r2_site_pct'] = (avg_flow / max_capacity * 100) if max_capacity > 0 else 0
            
            # Annual volume
            stats['annual_vol_r2_site_ML'] = avg_flow * days_per_year
        else:
            stats['utilization_r2_site_pct'] = 0
            stats['annual_vol_r2_site_ML'] = 0
    
    return stats


def extract_presets_from_historical(historical_results):
    """Extract preset parameters from last 12 months of historical data"""
    if historical_results is None or len(historical_results) == 0:
        return None
    
    # Check for required columns
    required_cols = ['r1_level_ML', 'r2_level_ML']
    available_cols = [col for col in required_cols if col in historical_results.columns]
    
    if len(available_cols) < 2:
        return None
    
    # Find rows where all required columns have valid (non-null) data
    valid_rows = historical_results[available_cols].notna().all(axis=1)
    
    if not valid_rows.any():
        return None
    
    # Get the last date where all required data is valid
    valid_data = historical_results[valid_rows]
    last_valid_date = valid_data.index.max()
    
    # Calculate 12 months ago from the last valid date (not from today)
    twelve_months_ago = last_valid_date - pd.Timedelta(days=365)
    
    # Get recent data (last 12 months from last valid date)
    recent_data = historical_results[
        (historical_results.index >= twelve_months_ago) & 
        (historical_results.index <= last_valid_date)
    ]
    
    if len(recent_data) == 0:
        return None
    
    # Get the last valid row for initial levels
    last_valid_row = valid_data.loc[last_valid_date]
    
    # Extract initial levels from last valid date
    r1_initial = last_valid_row['r1_level_ML']
    r2_initial = last_valid_row['r2_level_ML']
    
    # Validate that values are reasonable
    if pd.isna(r1_initial) or pd.isna(r2_initial):
        return None
    
    presets = {
        'r1_initial': int(r1_initial),
        'r2_initial': int(r2_initial),
        'pump_river_to_r1_avg': recent_data['pump_river_to_r1_ML_day'].mean() if 'pump_river_to_r1_ML_day' in recent_data.columns else 0,
        'pump_r1_to_r2_avg': recent_data['pump_r1_to_r2_ML_day'].mean() if 'pump_r1_to_r2_ML_day' in recent_data.columns else 0,
        'pump_r2_to_site_avg': recent_data['pump_r2_to_site_ML_day'].mean() if 'pump_r2_to_site_ML_day' in recent_data.columns else 0,
        'demand_avg': recent_data['pump_r2_to_site_ML_day'].mean() if 'pump_r2_to_site_ML_day' in recent_data.columns else 9.8,
        'model_start_date': last_valid_date + pd.Timedelta(days=1),
        'last_historical_date': last_valid_date
    }
    
    # Replace NaN values in averages with defaults
    if pd.isna(presets['pump_river_to_r1_avg']):
        presets['pump_river_to_r1_avg'] = 0
    if pd.isna(presets['pump_r1_to_r2_avg']):
        presets['pump_r1_to_r2_avg'] = 0
    if pd.isna(presets['pump_r2_to_site_avg']):
        presets['pump_r2_to_site_avg'] = 9.8
    if pd.isna(presets['demand_avg']):
        presets['demand_avg'] = 9.8
    
    presets['model_start_year'] = presets['model_start_date'].year
    presets['model_end_year'] = presets['model_start_year'] + 5
    
    return presets


def prepare_flow_scenario(base_path, drought_years, rain_years):
    """Load and modify flow scenario based on user selections"""
    from pathlib import Path
    
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
    """Insert extreme year template into target hydrological year"""
    flows = base_flows.copy()
    extreme_data = extreme_template.copy()

    def shift_date(d):
        if d.month >= 11:
            return d.replace(year=target_year - 1)
        else:
            return d.replace(year=target_year)

    extreme_data.index = extreme_data.index.map(shift_date)

    hydro_start = pd.Timestamp(f"{target_year - 1}-11-01")
    hydro_end = pd.Timestamp(f"{target_year}-10-31")

    flows_filtered = flows[(flows.index < hydro_start) | (flows.index > hydro_end)]
    result = pd.concat([flows_filtered, extreme_data]).sort_index()

    return result


# Missing import
import os


def prepare_prior_12_months_data(full_historical_data, current_results):
    """
    Calculate prior 12 months averages for horizontal reference lines.
    
    Args:
        full_historical_data: Full DataFrame with all historical data
        current_results: Filtered DataFrame for current viewing period
        
    Returns:
        Dict with average values from 12 months prior to viewing period
    """
    if full_historical_data is None or len(full_historical_data) == 0:
        return None
    if current_results is None or len(current_results) == 0:
        return None
    
    # Get the start date of current viewing period
    current_start = current_results.index.min()
    
    # Define 12 months prior period (365 days before current start, going back 365 days)
    prior_end = current_start - pd.Timedelta(days=1)
    prior_start = prior_end - pd.Timedelta(days=365)
    
    # Filter to prior 12 months period
    prior_mask = (full_historical_data.index >= prior_start) & (full_historical_data.index <= prior_end)
    prior_data = full_historical_data.loc[prior_mask]
    
    if len(prior_data) == 0:
        return None
    
    # Calculate averages for key columns
    averages = {}
    
    if 'r1_level_ML' in prior_data.columns:
        averages['r1_level_ML'] = prior_data['r1_level_ML'].mean()
    
    if 'r2_level_ML' in prior_data.columns:
        averages['r2_level_ML'] = prior_data['r2_level_ML'].mean()
    
    if 'pump_river_to_r1_ML_day' in prior_data.columns:
        averages['pump_river_to_r1_ML_day'] = prior_data['pump_river_to_r1_ML_day'].mean()
    
    if 'pump_r1_to_r2_ML_day' in prior_data.columns:
        averages['pump_r1_to_r2_ML_day'] = prior_data['pump_r1_to_r2_ML_day'].mean()
    
    if 'pump_r2_to_site_ML_day' in prior_data.columns:
        averages['pump_r2_to_site_ML_day'] = prior_data['pump_r2_to_site_ML_day'].mean()
    
    return averages

