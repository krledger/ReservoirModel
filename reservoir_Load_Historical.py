import pandas as pd
import numpy as np
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Construct the path to the Excel file
excel_path = script_dir / '2511_KL_model.xlsx'

# Verify the file exists
if not excel_path.exists():
    raise FileNotFoundError(f"Excel file not found at: {excel_path}")


def load_and_consolidate_historical_data_WEEKLY(excel_path):
    """
    Load historical reservoir data and consolidate into weekly format.
    FILLS IN MISSING WEEKS with interpolation.

    Returns:
        pd.DataFrame: Weekly data with gaps filled
    """
    print("Loading historical data...")

    # Load all sheets
    scd_df = pd.read_excel(excel_path, sheet_name='SCD')
    tnd_df = pd.read_excel(excel_path, sheet_name='TND')
    trailer_df = pd.read_excel(excel_path, sheet_name='Trailer pumps')
    booster_df = pd.read_excel(excel_path, sheet_name='Booster Pumps')
    raw_out_df = pd.read_excel(excel_path, sheet_name='SCD Raw out')

    # ============================================================================
    # 1. Process SCD (Surhs Creek Dam) = R2 levels
    # ============================================================================
    print("  Processing SCD (R2) data...")
    scd_df['date'] = pd.to_datetime(scd_df['Unnamed: 0'], errors='coerce')

    # Fix date errors
    bad_dates = scd_df[scd_df['date'].isna()]['Unnamed: 0']
    if len(bad_dates) > 0:
        print(f"    Warning: {len(bad_dates)} problematic dates found")
        for idx, bad_date in bad_dates.items():
            if isinstance(bad_date, str) and '5025' in bad_date:
                fixed = bad_date.replace('5025', '2025')
                scd_df.loc[idx, 'date'] = pd.to_datetime(fixed, errors='coerce')

    scd_df = scd_df[~scd_df['date'].isna()].copy()
    scd_df = scd_df[['date', 'SCD level %', 'SCD ML']].copy()
    scd_df.columns = ['date', 'r2_level_pct', 'r2_level_ML']
    scd_df = scd_df.sort_values('date')

    # ============================================================================
    # 2. Process TND (Turkeys Nest Dam) = R1 levels
    # ============================================================================
    print("  Processing TND (R1) data...")
    tnd_df['date'] = pd.to_datetime(tnd_df['Unnamed: 0'], errors='coerce')
    tnd_df = tnd_df[~tnd_df['date'].isna()].copy()
    tnd_df = tnd_df[['date', 'TND %', 'ML']].copy()
    tnd_df.columns = ['date', 'r1_level_pct', 'r1_level_ML']
    tnd_df = tnd_df.sort_values('date')

    # ============================================================================
    # 3. Process Trailer Pumps (River to R1) - FIXED
    # ============================================================================
    print("  Processing Trailer Pumps (River→R1) data...")
    trailer_df['date'] = pd.to_datetime(trailer_df['date'])
    trailer_df = trailer_df.sort_values('date')

    trailer_df['days_since_last'] = trailer_df['date'].diff().dt.days
    trailer_df['pump_river_to_r1_ML_day'] = (
            trailer_df['combined ML'] / trailer_df['days_since_last']
    )

    # Remove impossible rates
    trailer_df.loc[trailer_df['pump_river_to_r1_ML_day'] > 50, 'pump_river_to_r1_ML_day'] = np.nan
    trailer_df.loc[trailer_df['pump_river_to_r1_ML_day'] < 0, 'pump_river_to_r1_ML_day'] = np.nan

    trailer_processed = trailer_df[['date', 'pump_river_to_r1_ML_day']].copy()

    # ============================================================================
    # 4. Process Booster Pumps (R1 to R2) - FIXED
    # ============================================================================
    print("  Processing Booster Pumps (R1→R2) data...")
    booster_df['date'] = pd.to_datetime(booster_df['Unnamed: 0'])
    booster_df = booster_df.sort_values('date')

    booster_df['days_since_last'] = booster_df['date'].diff().dt.days
    booster_df['pump_r1_to_r2_ML_day'] = (
            booster_df['Meter 79'] / booster_df['days_since_last']
    )

    # Remove impossible rates
    booster_df.loc[booster_df['pump_r1_to_r2_ML_day'] > 20, 'pump_r1_to_r2_ML_day'] = np.nan
    booster_df.loc[booster_df['pump_r1_to_r2_ML_day'] < 0, 'pump_r1_to_r2_ML_day'] = np.nan

    booster_processed = booster_df[['date', 'pump_r1_to_r2_ML_day']].copy()

    # ============================================================================
    # 5. Process SCD Raw Out (R2 to Site = Demand) - FIXED
    # ============================================================================
    print("  Processing SCD Raw Out (R2→Site = Demand) data...")
    raw_out_df['date'] = pd.to_datetime(raw_out_df['Unnamed: 0'])
    raw_out_df = raw_out_df.sort_values('date')

    raw_out_df['days_since_last'] = raw_out_df['date'].diff().dt.days
    raw_out_df['pump_r2_to_site_ML_day'] = (
            raw_out_df['Meter 14 ML'] / raw_out_df['days_since_last']
    )

    # Remove impossible rates
    raw_out_df.loc[raw_out_df['pump_r2_to_site_ML_day'] > 20, 'pump_r2_to_site_ML_day'] = np.nan
    raw_out_df.loc[raw_out_df['pump_r2_to_site_ML_day'] < 0, 'pump_r2_to_site_ML_day'] = np.nan

    raw_out_processed = raw_out_df[['date', 'pump_r2_to_site_ML_day']].copy()

    # ============================================================================
    # 6. Merge all data
    # ============================================================================
    print("  Merging all datasets...")
    consolidated = scd_df.copy()

    consolidated = consolidated.merge(tnd_df, on='date', how='outer')
    consolidated = consolidated.merge(trailer_processed, on='date', how='outer')
    consolidated = consolidated.merge(booster_processed, on='date', how='outer')
    consolidated = consolidated.merge(raw_out_processed, on='date', how='outer')

    consolidated = consolidated.sort_values('date').reset_index(drop=True)

    # Check for and handle duplicate dates
    if consolidated['date'].duplicated().any():
        print(f"    Warning: {consolidated['date'].duplicated().sum()} duplicate dates found")
        print(f"    Averaging values for duplicate dates...")
        consolidated = consolidated.groupby('date', as_index=False).mean(numeric_only=True)
        consolidated = consolidated.sort_values('date')

    consolidated.set_index('date', inplace=True)

    # ============================================================================
    # 7. FILL IN MISSING WEEKS
    # ============================================================================
    print("\n  Filling in missing weeks...")

    # Create complete weekly date range
    start_date = consolidated.index.min()
    end_date = consolidated.index.max()

    # Generate weekly frequency (every 7 days)
    weekly_range = pd.date_range(start=start_date, end=end_date, freq='7D')

    print(f"    Original data: {len(consolidated)} records")
    print(f"    Expected weekly records: {len(weekly_range)}")
    print(f"    Missing weeks: {len(weekly_range) - len(consolidated)}")

    # Reindex to weekly
    consolidated_weekly = consolidated.reindex(weekly_range)

    # INTERPOLATION STRATEGY (for missing weeks only):
    # 1. Reservoir levels: Linear interpolation (up to 4 weeks gap)
    # 2. Pump rates: Forward fill (up to 2 weeks gap)

    # Interpolate reservoir levels
    level_cols = ['r1_level_ML', 'r2_level_ML', 'r1_level_pct', 'r2_level_pct']
    for col in level_cols:
        if col in consolidated_weekly.columns:
            # Linear interpolation (max 4 weeks = 28 days)
            consolidated_weekly[col] = consolidated_weekly[col].interpolate(
                method='time',
                limit=4
            )

    # Forward fill pump rates
    pump_cols = ['pump_river_to_r1_ML_day', 'pump_r1_to_r2_ML_day', 'pump_r2_to_site_ML_day']
    for col in pump_cols:
        if col in consolidated_weekly.columns:
            # Forward fill up to 2 weeks
            consolidated_weekly[col] = consolidated_weekly[col].ffill(limit=2)

    # ============================================================================
    # 8. Calculate derived metrics
    # ============================================================================
    consolidated_weekly['demand_supplied_ML_day'] = consolidated_weekly['pump_r2_to_site_ML_day']

    # Calculate storage changes
    consolidated_weekly['r1_storage_change_ML'] = consolidated_weekly['r1_level_ML'].diff()
    consolidated_weekly['r2_storage_change_ML'] = consolidated_weekly['r2_level_ML'].diff()

    # ============================================================================
    # 9. Add data quality flags
    # ============================================================================
    consolidated_weekly['is_original_measurement'] = consolidated_weekly.index.isin(consolidated.index)
    consolidated_weekly['has_r1_level'] = ~consolidated_weekly['r1_level_ML'].isna()
    consolidated_weekly['has_r2_level'] = ~consolidated_weekly['r2_level_ML'].isna()
    consolidated_weekly['has_river_pump'] = ~consolidated_weekly['pump_river_to_r1_ML_day'].isna()
    consolidated_weekly['has_r1_r2_pump'] = ~consolidated_weekly['pump_r1_to_r2_ML_day'].isna()
    consolidated_weekly['has_r2_site_pump'] = ~consolidated_weekly['pump_r2_to_site_ML_day'].isna()

    # ============================================================================
    # 10. Clean invalid values
    # ============================================================================
    print("\n  Cleaning invalid values...")
    errors_found = 0

    # Check reservoir levels
    if 'r1_level_ML' in consolidated_weekly.columns:
        invalid = (consolidated_weekly['r1_level_ML'] < 0) | (consolidated_weekly['r1_level_ML'] > 1300)
        if invalid.sum() > 0:
            print(f"    R1 level: {invalid.sum()} invalid values")
            consolidated_weekly.loc[invalid, 'r1_level_ML'] = np.nan
            errors_found += invalid.sum()

    if 'r2_level_ML' in consolidated_weekly.columns:
        invalid = (consolidated_weekly['r2_level_ML'] < 0) | (consolidated_weekly['r2_level_ML'] > 1380)
        if invalid.sum() > 0:
            print(f"    R2 level: {invalid.sum()} invalid values")
            consolidated_weekly.loc[invalid, 'r2_level_ML'] = np.nan
            errors_found += invalid.sum()

    if errors_found == 0:
        print(f"    ✓ No invalid values found")

    return consolidated_weekly


if __name__ == "__main__":
    print("=" * 80)
    print("HISTORICAL DATA CONSOLIDATION - WEEKLY WITH GAPS FILLED")
    print("=" * 80)
    print("\nFEATURES:")
    print("  • Treats Excel data as periodic volumes (not cumulative)")
    print("  • Fills in missing weeks using interpolation")
    print("  • Linear interpolation for reservoir levels (up to 4 weeks)")
    print("  • Forward fill for pump rates (up to 2 weeks)\n")

    historical_data = load_and_consolidate_historical_data_WEEKLY(excel_path)

    # Generate summary
    print("\n" + "=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)

    print(f"\nDate Range: {historical_data.index.min().date()} to {historical_data.index.max().date()}")
    print(f"Total Records: {len(historical_data)} (weekly frequency)")

    original_count = historical_data['is_original_measurement'].sum()
    interpolated_count = len(historical_data) - original_count
    print(f"\nData Composition:")
    print(f"  Original measurements: {original_count} ({original_count / len(historical_data) * 100:.1f}%)")
    print(f"  Filled missing weeks:  {interpolated_count} ({interpolated_count / len(historical_data) * 100:.1f}%)")

    print(f"\nData Coverage (after filling gaps):")
    print(
        f"  R2 (Surhs) levels:     {historical_data['has_r2_level'].sum()} weeks ({historical_data['has_r2_level'].sum() / len(historical_data) * 100:.1f}%)")
    print(
        f"  R1 (Turkeys) levels:   {historical_data['has_r1_level'].sum()} weeks ({historical_data['has_r1_level'].sum() / len(historical_data) * 100:.1f}%)")
    print(
        f"  River→R1 pump:         {historical_data['has_river_pump'].sum()} weeks ({historical_data['has_river_pump'].sum() / len(historical_data) * 100:.1f}%)")
    print(
        f"  R1→R2 pump:            {historical_data['has_r1_r2_pump'].sum()} weeks ({historical_data['has_r1_r2_pump'].sum() / len(historical_data) * 100:.1f}%)")
    print(
        f"  R2→Site pump (demand): {historical_data['has_r2_site_pump'].sum()} weeks ({historical_data['has_r2_site_pump'].sum() / len(historical_data) * 100:.1f}%)")

    print(f"\nKey Statistics (daily rates):")

    metrics = {
        'R2 Level (ML)': 'r2_level_ML',
        'R1 Level (ML)': 'r1_level_ML',
        'River→R1 (ML/day)': 'pump_river_to_r1_ML_day',
        'R1→R2 (ML/day)': 'pump_r1_to_r2_ML_day',
        'R2→Site DEMAND (ML/day)': 'pump_r2_to_site_ML_day'
    }

    for label, col in metrics.items():
        if col in historical_data.columns:
            data = historical_data[col].dropna()
            if len(data) > 0:
                print(f"\n  {label}:")
                print(f"    Mean:   {data.mean():>8.2f}")
                print(f"    Median: {data.median():>8.2f}")
                print(f"    Min:    {data.min():>8.2f}")
                print(f"    Max:    {data.max():>8.2f}")

    # Save consolidated data
    csv_path = script_dir / 'historical_data_consolidated.csv'
    historical_data.to_csv(csv_path, index_label='date')

    print("\n" + "=" * 80)
    print("✓ CONSOLIDATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput file: {csv_path}")
    print(f"Total records: {len(historical_data)} (weekly)")
    print(f"Date range: {historical_data.index.min().date()} to {historical_data.index.max().date()}")

    print("\n" + "=" * 80)
    print("Sample data (first 10 weeks):")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    sample_cols = ['r2_level_ML', 'r1_level_ML', 'pump_river_to_r1_ML_day',
                   'pump_r1_to_r2_ML_day', 'pump_r2_to_site_ML_day', 'is_original_measurement']
    print(historical_data[sample_cols].head(10))