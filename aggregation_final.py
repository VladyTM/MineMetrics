import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# === Script: Aggregate RDT truck metrics across all monthly files in a folder ===

def main():
    """
    Processes all Excel files in the 'monthly_data' folder (named like 'January 21.xlsx'),
    filters for RDT equipment, extracts distinct trucks,
    sums BCM, production hours, engine hours, loads (AF–AQ),
    computes weighted haulage distance,
    and outputs a consolidated Excel file with a progress bar.
    """
    # Configuration
    input_folder = Path('/Users/vlad/Documents/School Documents/CIPQ-2025/data/raw')  # folder containing monthly .xlsx files
    output_file = Path('Correct Totals.xlsx')

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Error: input folder '{input_folder}' not found or is not a directory.")
        return

    all_results = []
    files = sorted(input_folder.glob('*.xlsx'))
    if not files:
        print(f"No .xlsx files found in '{input_folder}'")
        return

    # Loop with progress bar
    for file in tqdm(files, desc='Aggregating files'):
        # Parse MonthYear from filename
        try:
            month_year_dt = datetime.strptime(file.stem, '%B %y').date()
        except ValueError:
            print(f"Skipping '{file.name}': filename does not match 'Month YY' format.")
            continue

        # Read raw data (rows 3–5000, no headers)
        df = pd.read_excel(file, header=None, skiprows=2, nrows=4998)

        # Extract columns
        equipment     = df.iloc[:, 23].astype(str).str.strip().str.upper()     # Column X
        trucks        = df.iloc[:, 24].astype(str).str.strip()                 # Column Y
        bcm_vals      = pd.to_numeric(df.iloc[:, 46], errors='coerce')         # Column AU
        distance_vals = pd.to_numeric(df.iloc[:, 17], errors='coerce')         # Column R
        load_cols     = df.iloc[:, 31:43].apply(pd.to_numeric, errors='coerce') # Columns AF–AQ
        engine_vals   = pd.to_numeric(df.iloc[:, 51], errors='coerce')         # Column AZ

        # Filter for RDT equipment
        is_rdt = (equipment == 'RDT')
        unique_trucks = pd.Series(trucks[is_rdt].unique())

        # Aggregate metrics per truck
        records = []
        for truck in unique_trucks:
            mask = is_rdt & (trucks == truck)
            bcm_sum      = bcm_vals[mask].sum(skipna=True)
            prod_hours   = load_cols[mask].gt(0).sum(axis=1).sum()
            engine_sum   = engine_vals[mask].sum(skipna=True)
            # Sum loads across AF–AQ for each truck
            loads_sum    = load_cols[mask].sum(axis=1, skipna=True).sum()
            weighted_sum = (distance_vals[mask] * bcm_vals[mask]).sum(skipna=True)
            weighted_avg = weighted_sum / bcm_sum if bcm_sum else np.nan

            records.append({
                'MonthYear':        month_year_dt,
                'Truck':            truck,
                'TotalBCM':         bcm_sum,
                'ProductionHours':  prod_hours,
                'EngineHours':      engine_sum,
                'TotalLoads':       loads_sum,
                'WeightedHaulDist': weighted_avg
            })

        # Append this month's results
        all_results.append(pd.DataFrame(records))

    # Combine all months
    final_df = pd.concat(all_results, ignore_index=True)

    # Save consolidated results
    final_df.to_excel(output_file, index=False)
    print(f"✅ Aggregated metrics for {len(files)} files written to {output_file}")

if __name__ == '__main__':
    main()
