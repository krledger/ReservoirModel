"""
Test script for reservoir simulation water balance verification
Tests core logic with simple, predictable inputs
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Import the ReservoirSystem class (assuming it's in the same directory)
import sys
sys.path.append('.')

# We'll recreate the minimal ReservoirSystem class here for testing
class TestReservoirSystem:
    """Minimal reservoir system for testing"""
    
    def __init__(self, config):
        self.config = config
        
    def simulate(self, climate_data, flow_data, params):
        """
        Simplified simulation for testing
        
        Args:
            climate_data: DataFrame with columns: temperature, humidity, precipitation
            flow_data: DataFrame with column: river_flow_ML_day
            params: dict with pump rates, demand, initial levels
        """
        
        # Join data
        data = climate_data.join(flow_data, how='inner')
        n = len(data)
        
        # Extract config
        r1_config = self.config['reservoirs'][0]
        r2_config = self.config['reservoirs'][1]
        
        # Initialise tracking arrays
        r1_level = np.zeros(n)
        r2_level = np.zeros(n)
        pump_river_to_r1 = np.zeros(n)
        pump_r1_to_r2 = np.zeros(n)
        pump_r2_to_site = np.zeros(n)
        evap_r1 = np.zeros(n)
        evap_r2 = np.zeros(n)
        seepage_r1 = np.zeros(n)
        seepage_r2 = np.zeros(n)
        pluvial_r1 = np.zeros(n)
        pluvial_r2 = np.zeros(n)
        
        # Initial levels
        r1_level[0] = params['r1_initial']
        r2_level[0] = params['r2_initial']
        
        # Fixed parameters
        pump_river_r1_max = params['pump_river_to_r1_max']
        pump_r1_r2_max = params['pump_r1_to_r2_max']
        pump_r2_site_max = params['pump_r2_to_site_max']
        demand = params['demand_ML_day']
        
        r1_capacity = r1_config['capacity_ML']
        r2_capacity = r2_config['capacity_ML']
        r1_min = r1_config.get('min_capacity_ML', 0)
        r2_min = r2_config.get('min_capacity_ML', 0)
        
        print("\n" + "="*80)
        print("SIMULATION CONFIGURATION")
        print("="*80)
        print(f"\nReservoir Capacities:")
        print(f"  R1 (Turkeys Nest): {r1_capacity} ML (min: {r1_min} ML)")
        print(f"  R2 (Surhs Creek): {r2_capacity} ML (min: {r2_min} ML)")
        print(f"\nInitial Levels:")
        print(f"  R1: {r1_level[0]} ML ({r1_level[0]/r1_capacity*100:.1f}%)")
        print(f"  R2: {r2_level[0]} ML ({r2_level[0]/r2_capacity*100:.1f}%)")
        print(f"\nPump Capacities:")
        print(f"  River → R1: {pump_river_r1_max} ML/day")
        print(f"  R1 → R2: {pump_r1_r2_max} ML/day")
        print(f"  R2 → Site: {pump_r2_site_max} ML/day")
        print(f"\nDemand: {demand} ML/day")
        print(f"\nSimulation Period: {n} days")
        print("="*80 + "\n")
        
        # Simulation loop
        for i in range(1, n):
            # Get current levels
            r1_current = r1_level[i-1]
            r2_current = r2_level[i-1]
            
            river_flow = data['river_flow_ML_day'].iloc[i]
            precip = data['precipitation'].iloc[i]
            temp = data['temperature'].iloc[i]
            
            # ================================================================
            # STEP 1: PUMP RIVER → R1
            # ================================================================
            available_capacity_r1 = r1_capacity - r1_current
            
            # For testing, assume river flow always allows pumping
            if available_capacity_r1 > 0:
                pump_river_to_r1[i] = min(pump_river_r1_max, available_capacity_r1)
            else:
                pump_river_to_r1[i] = 0
            
            # ================================================================
            # STEP 2: CALCULATE LOSSES AND INFLOWS
            # ================================================================
            # Simple evaporation (proportional to temperature)
            evap_rate_mm = temp * 0.2  # Simplified: 0.2mm per degree
            evap_r1[i] = (evap_rate_mm / 1000) * r1_config['surface_area_m²'] / 1000
            evap_r2[i] = (evap_rate_mm / 1000) * r2_config['surface_area_m²'] / 1000
            
            # Seepage (constant)
            seepage_r1[i] = r1_config['losses']['seepage_ML_day']
            seepage_r2[i] = r2_config['losses']['seepage_ML_day']
            
            # Pluvial inflow (rain on reservoir surface)
            if precip > 2.0:
                pluvial_r1[i] = (precip / 1000) * r1_config['surface_area_m²'] / 1000
                pluvial_r2[i] = (precip / 1000) * r2_config['surface_area_m²'] / 1000
            else:
                pluvial_r1[i] = 0
                pluvial_r2[i] = 0
            
            # ================================================================
            # STEP 3: UPDATE R1 WITH INFLOWS/OUTFLOWS
            # ================================================================
            r1_current += pump_river_to_r1[i]  # Inflow from river
            r1_current += pluvial_r1[i]  # Rain on surface
            r1_current -= seepage_r1[i]  # Seepage loss
            r1_current -= evap_r1[i]  # Evaporation loss
            
            # ================================================================
            # STEP 4: PUMP R1 → R2
            # ================================================================
            available_water_r1 = max(0, r1_current - r1_min)
            available_capacity_r2 = max(0, r2_capacity - r2_current)
            
            if available_water_r1 > 0 and available_capacity_r2 > 0:
                # Pump to meet R2 needs or shed R1 excess
                pump_r1_to_r2[i] = min(
                    pump_r1_r2_max,
                    available_water_r1,
                    available_capacity_r2
                )
            else:
                pump_r1_to_r2[i] = 0
            
            # Apply pump
            r1_current -= pump_r1_to_r2[i]
            r1_current = max(r1_min, min(r1_current, r1_capacity))
            r1_level[i] = r1_current
            
            # ================================================================
            # STEP 5: UPDATE R2 WITH INFLOWS/OUTFLOWS
            # ================================================================
            r2_current += pump_r1_to_r2[i]  # Inflow from R1
            r2_current += pluvial_r2[i]  # Rain on surface
            r2_current -= seepage_r2[i]  # Seepage loss
            r2_current -= evap_r2[i]  # Evaporation loss
            
            # ================================================================
            # STEP 6: PUMP R2 → SITE (MEET DEMAND)
            # ================================================================
            available_water_r2 = max(0, r2_current - r2_min)
            
            if available_water_r2 > 0:
                pump_r2_to_site[i] = min(
                    pump_r2_site_max,
                    demand,
                    available_water_r2
                )
            else:
                pump_r2_to_site[i] = 0
            
            # Apply pump
            r2_current -= pump_r2_to_site[i]
            r2_current = max(r2_min, min(r2_current, r2_capacity))
            r2_level[i] = r2_current
            
            # ================================================================
            # PRINT DETAILED OUTPUT FOR FIRST FEW DAYS
            # ================================================================
            if i <= 5:
                print(f"\n{'='*80}")
                print(f"DAY {i}")
                print(f"{'='*80}")
                print(f"\nStarting Levels:")
                print(f"  R1: {r1_level[i-1]:.1f} ML")
                print(f"  R2: {r2_level[i-1]:.1f} ML")
                
                print(f"\nR1 Water Balance:")
                print(f"  + Pump from River: {pump_river_to_r1[i]:.2f} ML")
                print(f"  + Rain on surface: {pluvial_r1[i]:.2f} ML")
                print(f"  - Seepage: {seepage_r1[i]:.2f} ML")
                print(f"  - Evaporation: {evap_r1[i]:.2f} ML")
                print(f"  - Pump to R2: {pump_r1_to_r2[i]:.2f} ML")
                r1_net = pump_river_to_r1[i] + pluvial_r1[i] - seepage_r1[i] - evap_r1[i] - pump_r1_to_r2[i]
                print(f"  = Net change: {r1_net:.2f} ML")
                print(f"  → Ending level: {r1_level[i]:.1f} ML")
                
                print(f"\nR2 Water Balance:")
                print(f"  + Pump from R1: {pump_r1_to_r2[i]:.2f} ML")
                print(f"  + Rain on surface: {pluvial_r2[i]:.2f} ML")
                print(f"  - Seepage: {seepage_r2[i]:.2f} ML")
                print(f"  - Evaporation: {evap_r2[i]:.2f} ML")
                print(f"  - Pump to Site: {pump_r2_to_site[i]:.2f} ML")
                r2_net = pump_r1_to_r2[i] + pluvial_r2[i] - seepage_r2[i] - evap_r2[i] - pump_r2_to_site[i]
                print(f"  = Net change: {r2_net:.2f} ML")
                print(f"  → Ending level: {r2_level[i]:.1f} ML")
                
                print(f"\nDemand Fulfilment:")
                print(f"  Demand: {demand:.1f} ML/day")
                print(f"  Supplied: {pump_r2_to_site[i]:.1f} ML/day")
                print(f"  Deficit: {demand - pump_r2_to_site[i]:.1f} ML/day")
        
        # Create results dataframe
        results = pd.DataFrame({
            'r1_level_ML': r1_level,
            'r2_level_ML': r2_level,
            'pump_river_to_r1': pump_river_to_r1,
            'pump_r1_to_r2': pump_r1_to_r2,
            'pump_r2_to_site': pump_r2_to_site,
            'evap_r1': evap_r1,
            'evap_r2': evap_r2,
            'seepage_r1': seepage_r1,
            'seepage_r2': seepage_r2,
            'pluvial_r1': pluvial_r1,
            'pluvial_r2': pluvial_r2,
            'demand_deficit': demand - pump_r2_to_site
        }, index=data.index)
        
        return results


def run_test():
    """Run simulation test with controlled inputs"""
    
    # ========================================================================
    # TEST CONFIGURATION
    # ========================================================================
    
    # Simple reservoir system config
    config = {
        'reservoirs': [
            {
                'name': 'Turkeys Nest',
                'capacity_ML': 1300,
                'min_capacity_ML': 150,
                'surface_area_m²': 180000,
                'losses': {'seepage_ML_day': 0.3}
            },
            {
                'name': 'Surhs Creek',
                'capacity_ML': 1380,
                'min_capacity_ML': 180,
                'surface_area_m²': 190000,
                'losses': {'seepage_ML_day': 0.4}
            }
        ]
    }
    
    # Simulation parameters
    params = {
        'r1_initial': 1000,  # Start at 77% capacity
        'r2_initial': 1100,  # Start at 80% capacity
        'pump_river_to_r1_max': 30,  # ML/day
        'pump_r1_to_r2_max': 10,  # ML/day
        'pump_r2_to_site_max': 12,  # ML/day (higher than demand to test constraint)
        'demand_ML_day': 9.8  # ML/day
    }
    
    # Create simple test data (30 days)
    dates = pd.date_range('2025-01-01', periods=30, freq='D')
    
    # Scenario 1: Constant conditions (no stress)
    climate_data = pd.DataFrame({
        'temperature': [25.0] * 30,  # Constant 25°C
        'humidity': [60.0] * 30,  # Constant humidity
        'precipitation': [0.0] * 30  # No rain
    }, index=dates)
    
    flow_data = pd.DataFrame({
        'river_flow_ML_day': [1000.0] * 30  # Constant high flow (always allows pumping)
    }, index=dates)
    
    # ========================================================================
    # RUN TEST
    # ========================================================================
    
    print("\n" + "="*80)
    print("RESERVOIR SIMULATION TEST")
    print("="*80)
    print("\nTEST SCENARIO: Steady state with constant demand")
    print("  - No rainfall")
    print("  - Constant temperature (25°C)")
    print("  - Constant river flow (high)")
    print("  - Constant demand (9.8 ML/day)")
    print("  - Both reservoirs start above 75% capacity")
    
    system = TestReservoirSystem(config)
    results = system.simulate(climate_data, flow_data, params)
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SIMULATION SUMMARY (30 DAYS)")
    print("="*80)
    
    print(f"\nReservoir Level Changes:")
    print(f"  R1 Start: {results['r1_level_ML'].iloc[0]:.1f} ML")
    print(f"  R1 End: {results['r1_level_ML'].iloc[-1]:.1f} ML")
    print(f"  R1 Change: {results['r1_level_ML'].iloc[-1] - results['r1_level_ML'].iloc[0]:.1f} ML")
    print(f"  R1 Daily avg change: {(results['r1_level_ML'].iloc[-1] - results['r1_level_ML'].iloc[0])/30:.2f} ML/day")
    
    print(f"\n  R2 Start: {results['r2_level_ML'].iloc[0]:.1f} ML")
    print(f"  R2 End: {results['r2_level_ML'].iloc[-1]:.1f} ML")
    print(f"  R2 Change: {results['r2_level_ML'].iloc[-1] - results['r2_level_ML'].iloc[0]:.1f} ML")
    print(f"  R2 Daily avg change: {(results['r2_level_ML'].iloc[-1] - results['r2_level_ML'].iloc[0])/30:.2f} ML/day")
    
    print(f"\nAverage Daily Flows:")
    print(f"  River → R1: {results['pump_river_to_r1'].mean():.2f} ML/day")
    print(f"  R1 → R2: {results['pump_r1_to_r2'].mean():.2f} ML/day")
    print(f"  R2 → Site: {results['pump_r2_to_site'].mean():.2f} ML/day")
    
    print(f"\nAverage Daily Losses:")
    print(f"  R1 Evaporation: {results['evap_r1'].mean():.2f} ML/day")
    print(f"  R1 Seepage: {results['seepage_r1'].mean():.2f} ML/day")
    print(f"  R2 Evaporation: {results['evap_r2'].mean():.2f} ML/day")
    print(f"  R2 Seepage: {results['seepage_r2'].mean():.2f} ML/day")
    total_losses = (results['evap_r1'].mean() + results['seepage_r1'].mean() + 
                   results['evap_r2'].mean() + results['seepage_r2'].mean())
    print(f"  Total System Losses: {total_losses:.2f} ML/day")
    
    print(f"\nDemand Fulfilment:")
    print(f"  Target Demand: {params['demand_ML_day']:.1f} ML/day")
    print(f"  Actual Supplied: {results['pump_r2_to_site'].mean():.2f} ML/day")
    print(f"  Average Deficit: {results['demand_deficit'].mean():.2f} ML/day")
    print(f"  Days with deficit: {(results['demand_deficit'] > 0.01).sum()}")
    
    # ========================================================================
    # WATER BALANCE CHECK
    # ========================================================================
    
    print("\n" + "="*80)
    print("WATER BALANCE VERIFICATION")
    print("="*80)
    
    # Total system water balance
    total_inflow = results['pump_river_to_r1'].sum() + results['pluvial_r1'].sum() + results['pluvial_r2'].sum()
    total_outflow = results['pump_r2_to_site'].sum()
    total_losses = (results['evap_r1'].sum() + results['seepage_r1'].sum() + 
                   results['evap_r2'].sum() + results['seepage_r2'].sum())
    
    storage_change = ((results['r1_level_ML'].iloc[-1] + results['r2_level_ML'].iloc[-1]) - 
                     (results['r1_level_ML'].iloc[0] + results['r2_level_ML'].iloc[0]))
    
    water_balance = total_inflow - total_outflow - total_losses - storage_change
    
    print(f"\nTotal Inflows: {total_inflow:.2f} ML")
    print(f"  - From river: {results['pump_river_to_r1'].sum():.2f} ML")
    print(f"  - From rain: {(results['pluvial_r1'].sum() + results['pluvial_r2'].sum()):.2f} ML")
    
    print(f"\nTotal Outflows: {total_outflow:.2f} ML")
    print(f"  - To site (demand): {results['pump_r2_to_site'].sum():.2f} ML")
    
    print(f"\nTotal Losses: {total_losses:.2f} ML")
    print(f"  - Evaporation: {(results['evap_r1'].sum() + results['evap_r2'].sum()):.2f} ML")
    print(f"  - Seepage: {(results['seepage_r1'].sum() + results['seepage_r2'].sum()):.2f} ML")
    
    print(f"\nStorage Change: {storage_change:.2f} ML")
    print(f"  - R1 change: {results['r1_level_ML'].iloc[-1] - results['r1_level_ML'].iloc[0]:.2f} ML")
    print(f"  - R2 change: {results['r2_level_ML'].iloc[-1] - results['r2_level_ML'].iloc[0]:.2f} ML")
    
    print(f"\n{'='*80}")
    print(f"WATER BALANCE ERROR: {water_balance:.4f} ML")
    print(f"  (Inflows - Outflows - Losses - Storage Change)")
    print(f"  Should be ~0.00 for correct mass balance")
    print(f"{'='*80}")
    
    if abs(water_balance) < 0.1:
        print("✓ PASS: Water balance is correct (error < 0.1 ML)")
    else:
        print("✗ FAIL: Water balance error is too large!")
    
    # ========================================================================
    # EXPECTED BEHAVIOUR CHECK
    # ========================================================================
    
    print("\n" + "="*80)
    print("EXPECTED BEHAVIOUR CHECK")
    print("="*80)
    
    print("\nExpected steady-state behaviour:")
    print("  1. River pumps ~30 ML/day to R1")
    print("  2. R1 transfers water to R2 to maintain balance")
    print("  3. R2 supplies demand of 9.8 ML/day")
    print("  4. System loses ~1-2 ML/day to evap/seepage")
    print("  5. R1 should gradually fill (inflow > outflow)")
    print("  6. R2 should stay relatively stable")
    
    # Check pump behaviour
    river_pump_avg = results['pump_river_to_r1'].mean()
    r1_r2_pump_avg = results['pump_r1_to_r2'].mean()
    r2_site_pump_avg = results['pump_r2_to_site'].mean()
    
    checks_passed = 0
    checks_total = 6
    
    print("\nActual behaviour:")
    if 25 <= river_pump_avg <= 30:
        print(f"  ✓ River pump: {river_pump_avg:.1f} ML/day (expected ~30)")
        checks_passed += 1
    else:
        print(f"  ✗ River pump: {river_pump_avg:.1f} ML/day (expected ~30)")
    
    if 8 <= r1_r2_pump_avg <= 11:
        print(f"  ✓ R1→R2 pump: {r1_r2_pump_avg:.1f} ML/day (expected ~10)")
        checks_passed += 1
    else:
        print(f"  ✗ R1→R2 pump: {r1_r2_pump_avg:.1f} ML/day (expected ~10)")
    
    if abs(r2_site_pump_avg - 9.8) < 0.5:
        print(f"  ✓ R2→Site pump: {r2_site_pump_avg:.1f} ML/day (expected 9.8)")
        checks_passed += 1
    else:
        print(f"  ✗ R2→Site pump: {r2_site_pump_avg:.1f} ML/day (expected 9.8)")
    
    if 1.0 <= total_losses/30 <= 2.5:
        print(f"  ✓ System losses: {total_losses/30:.1f} ML/day (expected 1-2)")
        checks_passed += 1
    else:
        print(f"  ✗ System losses: {total_losses/30:.1f} ML/day (expected 1-2)")
    
    r1_change_per_day = (results['r1_level_ML'].iloc[-1] - results['r1_level_ML'].iloc[0])/30
    if r1_change_per_day > 5:
        print(f"  ✓ R1 filling: +{r1_change_per_day:.1f} ML/day (expected positive)")
        checks_passed += 1
    else:
        print(f"  ✗ R1 filling: {r1_change_per_day:.1f} ML/day (expected positive)")
    
    r2_change_per_day = abs((results['r2_level_ML'].iloc[-1] - results['r2_level_ML'].iloc[0])/30)
    if r2_change_per_day < 3:
        print(f"  ✓ R2 stable: {r2_change_per_day:.1f} ML/day change (expected <3)")
        checks_passed += 1
    else:
        print(f"  ✗ R2 changing: {r2_change_per_day:.1f} ML/day (expected stable)")
    
    print(f"\n{'='*80}")
    print(f"BEHAVIOUR CHECKS: {checks_passed}/{checks_total} passed")
    print(f"{'='*80}")
    
    if checks_passed == checks_total:
        print("\n✓✓✓ ALL TESTS PASSED - Model logic appears correct! ✓✓✓")
    elif checks_passed >= checks_total * 0.8:
        print("\n⚠ MOST TESTS PASSED - Minor issues detected")
    else:
        print("\n✗✗✗ MULTIPLE TESTS FAILED - Model logic needs review ✗✗✗")
    
    return results


if __name__ == "__main__":
    results = run_test()
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nResults saved to memory for further inspection if needed.")
    print("You can examine 'results' DataFrame for detailed day-by-day data.")
