"""
Core reservoir operations and simulation logic

Losses come from ReservoirPhysics: FAO-56 open water evaporation on a surface
area that follows the stored volume, and Darcy-scaled seepage with a share
recovered and pumped back.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

import ReservoirPhysics as rp


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
        """Load climate for a named source (see ModelInputs.CLIMATE_SOURCES).

        Reads metricsDataFiles/<scenario>/raw_daily.parquet explicitly.  The previous
        loader walked metricsDataFiles and took the first raw_daily.parquet it met,
        so every scenario loaded the same file.
        """
        from ModelInputs import load_climate
        self.climate_data = load_climate(scenario)
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

    def calculate_evaporation(self, temperature, humidity, wind_speed=2.0, dtr=None, dates=None):
        """Open water evaporation, mm/day: FAO-56 reference evapotranspiration times the open water factor.

        temperature is the daily mean (degC), humidity specific humidity (g/kg), wind at 10 m (m/s),
        dtr the daily temperature range (degC) and dates the days, for solar radiation.
        """
        temperature = np.asarray(temperature, dtype=float)
        if np.ndim(wind_speed) == 0:
            wind_speed = np.full(len(temperature), float(wind_speed))
        doy = pd.DatetimeIndex(dates).dayofyear.values if dates is not None else np.full(len(temperature), 182)
        settings = rp.evaporation_settings(self.system_config or {})
        return rp.open_water_evaporation_mm(temperature, dtr, humidity, wind_speed, doy, settings)

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

        dtr = data['dtr_Ravenswood_degC'].values if 'dtr_Ravenswood_degC' in data.columns else None
        relative_humidity = self.calculate_relative_humidity(temperature, specific_humidity)
        evap_settings = rp.evaporation_settings(self.system_config)
        if params.get('open_water_factor') is not None:
            evap_settings['open_water_factor'] = float(params['open_water_factor'])
        evaporation_mm = rp.open_water_evaporation_mm(temperature, dtr, specific_humidity, wind,
                                                      data.index.dayofyear.values, evap_settings)

        # Get configuration
        reservoirs = self.system_config['system']['reservoirs']
        pumps = self.system_config['system']['pumps']
        r1_turkeys = reservoirs[0]
        r2_surhs = reservoirs[1]

        r1_min_capacity = params.get('r1_min_capacity', r1_turkeys.get('min_capacity_ML', 0))
        area_how = rp.area_method(self.system_config, params.get('area_method'))
        seep_override = params.get('seepage', {}) or {}
        r1_seep = rp.seepage_settings(r1_turkeys, seep_override.get('r1'))
        r2_seep = rp.seepage_settings(r2_surhs, seep_override.get('r2'))
        r1_curve = rp.area_curve(r1_turkeys, area_how)
        r2_curve = rp.area_curve(r2_surhs, area_how)
        r1_fixed = None if r1_curve else rp.fixed_area_m2(r1_turkeys)
        r2_fixed = None if r2_curve else rp.fixed_area_m2(r2_surhs)

        def area_at(curve, fixed, volume):
            return fixed if curve is None else float(np.interp(volume, curve[0], curve[1])) * 1e4

        def seep_at(res, s, volume):
            frac = min(max(volume / res['capacity_ML'], 0.0), 1.0)
            gross = s['gross_ML_day'] * frac ** s['exponent']
            return gross, gross * s['returned_pct'] / 100
        r2_min_capacity = params.get('r2_min_capacity', r2_surhs.get('min_capacity_ML', 0))
        # Community reserve on SCD: below it the site stops and only the water treatment plant draws,
        # down to the minimum pumping level
        r2_reserve = params.get('r2_community_reserve',
                                r2_surhs['capacity_ML'] * r2_surhs.get('community_reserve_pct', 0) / 100)
        wtp_ML_day = params.get('wtp_demand_ML_day', self.system_config['system'].get('wtp_demand_ML_day', 0.0))
        # Staged demand restrictions (off unless asked for): once the river has been below the extraction
        # trigger for trigger_after_days, site demand is cut as combined storage falls through each stage;
        # stages lift as storage recovers while pumping is possible.  The treatment plant is not cut.
        restr = dict(self.system_config['system'].get('restrictions', {}) or {})
        restr.update(params.get('restrictions', {}) or {})
        apply_restr = bool(params.get('apply_restrictions', False))
        stages = sorted(((st['below_pct'], st['reduce_pct']) for st in restr.get('stages', [])), reverse=True)
        restr_after = int(restr.get('trigger_after_days', 14))
        cap_total = r1_turkeys['capacity_ML'] + r2_surhs['capacity_ML']

        def stage_for(total):
            k = 0
            for j, (below, _) in enumerate(stages, start=1):
                if total < below / 100 * cap_total:
                    k = j
            return k

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
        r1_area_ha = np.zeros(n)
        r2_area_ha = np.zeros(n)
        r1_seep_gross = np.zeros(n)
        r1_seep_returned = np.zeros(n)
        r2_seep_gross = np.zeros(n)
        r2_seep_returned = np.zeros(n)
        demand_supplied = np.zeros(n)
        demand_deficit = np.zeros(n)
        site_supplied = np.zeros(n)
        wtp_supplied = np.zeros(n)
        site_deficit = np.zeros(n)
        wtp_deficit = np.zeros(n)
        restriction_stage = np.zeros(n, dtype=int)
        outside_days = 0
        stage = 0

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

        # Operating rules.  Every key is optional; when a key is absent the
        # original behaviour applies, so older configurations run unchanged.
        river_ops = dict(pumps[0].get('operations', {}))
        river_ops.update(params.get('river_operations', {}))
        withdraw_above = river_ops.get('withdraw_above_ML_day')          # flood withdrawal of the river pumps
        reinstate_below = river_ops.get('reinstate_below_ML_day', withdraw_above)
        reinstate_after = int(river_ops.get('reinstate_after_days', 0))
        tnd_stop_pct = river_ops.get('tnd_stop_pct')                       # stop filling TND at this level
        tnd_restart_pct = river_ops.get('tnd_restart_pct', tnd_stop_pct)   # resume below this level

        transfer_ops = dict(pumps[1].get('operations', {}))
        transfer_ops.update(params.get('transfer_operations', {}))
        transfer_control = transfer_ops.get('control', 'legacy')           # 'legacy', 'scd_target' or 'scd_hold'
        scd_target_pct = transfer_ops.get('scd_target_pct', 70)            # SCD operating level, % of capacity
        scd_stop_pct = transfer_ops.get('scd_stop_pct')                    # booster off at this SCD level
        offtakes_ML_day = transfer_ops.get('offtakes_ML_day', 0.0)         # taken from the transfer main before SCD

        demand_ML_day = params.get('demand_ML_day', 9.8)
        demand_series = params.get('demand_series')                        # optional daily forcing (hindcasts)
        if demand_series is not None:
            demand_base = pd.Series(demand_series).reindex(data.index).ffill().bfill().fillna(demand_ML_day).values
        else:
            demand_base = np.full(n, demand_ML_day, dtype=float)
        river_rate_series = params.get('river_pump_rate_series')           # optional daily pump capacity (hindcasts)
        if river_rate_series is not None:
            river_rate = pd.Series(river_rate_series).reindex(data.index).ffill().bfill().fillna(pump_river_to_r1_max).values
        else:
            river_rate = np.full(n, pump_river_to_r1_max, dtype=float)

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

        river_pumps_out = np.zeros(n, dtype=bool)
        offtake = np.zeros(n)
        withdrawn = False
        calm_days = 0
        tnd_full = False
        booster_off = False
        r1_area_ha[0] = area_at(r1_curve, r1_fixed, r1_level[0]) / 1e4
        r2_area_ha[0] = area_at(r2_curve, r2_fixed, r2_level[0]) / 1e4

        # Simulation loop
        for i in range(1, n):
            r1_current = r1_level[i - 1]
            r2_current = r2_level[i - 1]

            # Surface area and seepage follow the volume stored at the start of the day
            area_r1 = area_at(r1_curve, r1_fixed, r1_current)
            area_r2 = area_at(r2_curve, r2_fixed, r2_current)
            r1_area_ha[i], r2_area_ha[i] = area_r1 / 1e4, area_r2 / 1e4
            r1_seep_gross[i], r1_seep_returned[i] = seep_at(r1_turkeys, r1_seep, r1_current)
            r2_seep_gross[i], r2_seep_returned[i] = seep_at(r2_surhs, r2_seep, r2_current)
            r1_seep_net = r1_seep_gross[i] - r1_seep_returned[i]
            r2_seep_net = r2_seep_gross[i] - r2_seep_returned[i]

            # Flood withdrawal: pumps come out above the trigger and return once flow
            # has stayed below the reinstatement level for the stated number of days
            if withdraw_above is not None:
                if river_flow[i] > withdraw_above:
                    withdrawn = True
                    calm_days = 0
                elif withdrawn:
                    calm_days = calm_days + 1 if river_flow[i] < reinstate_below else 0
                    if calm_days >= reinstate_after:
                        withdrawn = False
            river_pumps_out[i] = withdrawn

            # Restriction stage, from storage at the start of the day
            # Days the river has been below the extraction trigger; flood withdrawal does not count, as there is
            # no shortage of river water then
            in_window = (not withdrawn) and low_cutoff <= river_flow[i] <= high_cutoff
            outside_days = outside_days + 1 if river_flow[i] < low_cutoff else 0
            if apply_restr and stages:
                now = stage_for(r1_current + r2_current)
                stage = max(stage, now) if outside_days >= restr_after else (min(stage, now) if in_window else stage)
                restriction_stage[i] = stage

            # TND fill control with hysteresis
            if tnd_stop_pct is not None:
                if r1_current >= r1_turkeys['capacity_ML'] * tnd_stop_pct / 100:
                    tnd_full = True
                elif r1_current <= r1_turkeys['capacity_ML'] * tnd_restart_pct / 100:
                    tnd_full = False

            # Pump: River to R1
            available_capacity_r1 = r1_turkeys['capacity_ML'] - r1_current
            if available_capacity_r1 <= 0 or withdrawn or tnd_full:
                pump_river_to_r1[i] = 0
            elif river_flow[i] >= low_cutoff and river_flow[i] <= high_cutoff:
                actual_pump_max = river_rate[i] * pump_river_r1_multipliers[i]
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
                pluvial_inflow_r1[i] = (precipitation[i] / 1000) * area_r1 / 1000
                pluvial_inflow_r2[i] = (precipitation[i] / 1000) * area_r2 / 1000
            else:
                pluvial_inflow_r1[i] = 0
                pluvial_inflow_r2[i] = 0

            total_inflow[i] = fluvial_inflow[i] + pluvial_inflow_r2[i]

            # Evaporation losses
            r1_evap_loss[i] = (evaporation_mm[i] / 1000) * area_r1 / 1000
            r2_evap_loss[i] = (evaporation_mm[i] / 1000) * area_r2 / 1000

            # Update R1
            r1_current += pump_river_to_r1[i] + pluvial_inflow_r1[i]
            r1_current -= r1_seep_net + r1_evap_loss[i]

            # Pump: R1 to R2
            available_water_r1 = max(0, r1_current - r1_min_capacity)
            available_capacity_r2 = max(0, r2_surhs['capacity_ML'] - r2_current)
            actual_pump_r1_r2_max = pump_r1_to_r2_max * pump_r1_r2_multipliers[i]
            actual_demand = demand_base[i] * demand_multipliers[i]

            if available_water_r1 <= 0 or available_capacity_r2 <= 0:
                pump_r1_to_r2[i] = 0
            elif transfer_control in ('scd_target', 'scd_hold'):
                # Booster hysteresis on SCD: off at or above the stop level, back on
                # once SCD has been drawn down to the operating level.
                if scd_stop_pct is not None:
                    if r2_current >= r2_surhs['capacity_ML'] * scd_stop_pct / 100:
                        booster_off = True
                    elif r2_current <= r2_surhs['capacity_ML'] * scd_target_pct / 100:
                        booster_off = False
                draw = (min(actual_demand, pump_r2_to_site_max)
                        + r2_seep_net + r2_evap_loss[i])
                if booster_off:
                    pump_r1_to_r2[i] = 0
                elif transfer_control == 'scd_hold':
                    # Replace the day's draw on SCD, so SCD holds its level and
                    # moves only on catchment runoff
                    pump_r1_to_r2[i] = min(actual_pump_r1_r2_max, draw + offtakes_ML_day, available_water_r1)
                else:
                    # Fill SCD towards the operating level
                    r2_projected = r2_current + fluvial_inflow[i] + pluvial_inflow_r2[i] - draw
                    gap = r2_surhs['capacity_ML'] * scd_target_pct / 100 - r2_projected
                    pump_r1_to_r2[i] = min(actual_pump_r1_r2_max, max(0.0, gap + offtakes_ML_day), available_water_r1)
            else:
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

            offtake[i] = min(offtakes_ML_day, pump_r1_to_r2[i])
            r1_current -= pump_r1_to_r2[i]
            r1_current = max(r1_min_capacity, min(r1_current, r1_turkeys['capacity_ML']))
            r1_level[i] = r1_current

            # Update R2
            r2_current += pump_r1_to_r2[i] - offtake[i] + fluvial_inflow[i] + pluvial_inflow_r2[i]
            r2_current -= r2_seep_net + r2_evap_loss[i]

            # Pump: R2 to Site.  The water treatment plant is supplied first and down to the minimum
            # pumping level; the rest of site only while SCD stays above the community reserve.
            actual_pump_r2_site_max = pump_r2_to_site_max * pump_r2_site_multipliers[i]
            wtp_need = min(wtp_ML_day, actual_demand)
            site_need = actual_demand - wtp_need
            if restriction_stage[i]:
                site_need *= 1 - stages[restriction_stage[i] - 1][1] / 100
            wtp_supplied[i] = max(0.0, min(wtp_need, actual_pump_r2_site_max, r2_current - r2_min_capacity))
            site_supplied[i] = max(0.0, min(site_need, actual_pump_r2_site_max - wtp_supplied[i],
                                            r2_current - wtp_supplied[i] - max(r2_reserve, r2_min_capacity)))
            pump_r2_to_site[i] = wtp_supplied[i] + site_supplied[i]
            r2_current -= pump_r2_to_site[i]

            wtp_deficit[i] = wtp_need - wtp_supplied[i]
            site_deficit[i] = site_need - site_supplied[i]
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
            'r1_area_ha': r1_area_ha,
            'r2_area_ha': r2_area_ha,
            'r1_seepage_gross_ML_day': r1_seep_gross,
            'r1_seepage_returned_ML_day': r1_seep_returned,
            'r1_seepage_ML_day': r1_seep_gross - r1_seep_returned,
            'r2_seepage_gross_ML_day': r2_seep_gross,
            'r2_seepage_returned_ML_day': r2_seep_returned,
            'r2_seepage_ML_day': r2_seep_gross - r2_seep_returned,
            'pump_river_to_r1_ML_day': pump_river_to_r1,
            'pump_r1_to_r2_ML_day': pump_r1_to_r2,
            'pump_r2_to_site_ML_day': pump_r2_to_site,
            'r1_level_ML': r1_level,
            'r2_level_ML': r2_level,
            'demand_supplied_ML': demand_supplied,
            'demand_deficit_ML': demand_deficit,
            'site_supplied_ML': site_supplied,
            'wtp_supplied_ML': wtp_supplied,
            'site_deficit_ML': site_deficit,
            'wtp_deficit_ML': wtp_deficit,
            'restriction_stage': restriction_stage,
            'offtake_ML_day': offtake,
            'river_pumps_withdrawn': river_pumps_out
        })
        results.set_index('date', inplace=True)
        return results


def operating_profile_params(config, profile='as_operated'):
    """Simulation params for a named operating profile in reservoir_system.json.

    'as_operated' (or a missing profile) returns no overrides, so the pump
    operations blocks in the configuration apply as calibrated.  Other profiles
    override pump rates and operating rules.
    """
    spec = config.get('operating_profiles', {}).get(profile, {}) or {}
    params = {}
    if spec.get('river_rate_ML_day') is not None:
        params['pump_river_to_r1_max'] = spec['river_rate_ML_day']
    if spec.get('transfer_rate_ML_day') is not None:
        params['pump_r1_to_r2_max'] = spec['transfer_rate_ML_day']
    if spec.get('river_operations'):
        params['river_operations'] = dict(spec['river_operations'])
    if spec.get('transfer_operations'):
        params['transfer_operations'] = dict(spec['transfer_operations'])
    return params


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
        'deficit_days': (results['demand_deficit_ML'] > 0.01).sum(),
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
    for use in ('site', 'wtp'):
        col = f'{use}_deficit_ML'
        if col in results.columns:
            stats[f'{use}_deficit'] = results[col].sum()
            stats[f'{use}_deficit_days'] = int((results[col] > 0.01).sum())
    
    # Calculate days of storage for each reservoir and combined
    # Get seepage and evaporation rates
    # Net seepage (gross less returned) as simulated; configured value at the mean level for older results
    if 'r1_seepage_ML_day' in results.columns:
        r1_seepage = results['r1_seepage_ML_day'].iloc[1:].mean()
        r2_seepage = results['r2_seepage_ML_day'].iloc[1:].mean()
    else:
        r1_seepage = float(rp.seepage_ML_day(reservoirs[0], stats['avg_r1'])[2])
        r2_seepage = float(rp.seepage_ML_day(reservoirs[1], stats['avg_r2'])[2])
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
    
    # Demand split: water treatment plant (meter 166, supplied from 014) and the rest of site
    wtp = recent_data['wtp_ML_day'].mean() if 'wtp_ML_day' in recent_data.columns else float('nan')
    presets['wtp_avg'] = 0.5 if pd.isna(wtp) else float(wtp)
    presets['site_avg'] = max(0.0, presets['demand_avg'] - presets['wtp_avg'])
    # Site demand by use (SiteActuals.DEMAND_USES and the unmetered remainder of 014)
    for use in ('plant', 'gland', 'dust', 'minor', 'other'):
        col = f'{use}_ML_day'
        v = recent_data[col].mean() if col in recent_data.columns else float('nan')
        presets[f'{use}_avg'] = None if pd.isna(v) else max(0.0, float(v))
    offtakes = recent_data['offtake_ML_day'].mean() if 'offtake_ML_day' in recent_data.columns else float('nan')
    presets['offtake_avg'] = None if pd.isna(offtakes) else float(offtakes)

    presets['model_start_year'] = presets['model_start_date'].year
    presets['model_end_year'] = presets['model_start_year'] + 5
    
    return presets


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

