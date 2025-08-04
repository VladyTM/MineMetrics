import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# === Configuration ===
input_folder = Path('/Users/vlad/Documents/School Documents/CIPQ-2025/data/raw')  # Folder containing the 53 monthly .xlsx files
output_file  = 'aggregated_results.xlsx'

# === Helper functions ===
def parse_monthyear_from_filename(filename: str) -> datetime.date:
    # Expects 'Month YY' as stem
    return datetime.strptime(Path(filename).stem, '%B %y').date()

# === Main processing ===
all_months = []
files = sorted(input_folder.glob('*.xlsx'))
for file in tqdm(files, desc='Aggregating files'):
    # 1. Extract month-year
    month_year = parse_monthyear_from_filename(file.name)

    # 2. Read raw data
    df = pd.read_excel(file, header=None, skiprows=2, nrows=4998)
    raw_type     = df.iloc[:, 23]
    raw_truck    = df.iloc[:, 24]
    bcm_raw      = df.iloc[:, 44]
    hour_raw     = df.iloc[:, 51]
    load_cols    = df.iloc[:, 31:43]
    distance_raw = df.iloc[:, 17]

    # 3. Clean & convert
    clean_type   = raw_type.astype(str).str.strip().str.upper()
    clean_truck  = raw_truck.astype(str).str.strip().str.upper()
    bcm_vals     = pd.to_numeric(bcm_raw, errors='coerce')
    hour_vals    = pd.to_numeric(hour_raw, errors='coerce')
    load_numeric = load_cols.apply(pd.to_numeric, errors='coerce')
    distance_vals= pd.to_numeric(distance_raw, errors='coerce')

    # 4. Compute production hours per shift
    prod_hours = load_numeric.gt(0).sum(axis=1)

    # 5. Truncate to consistent length
    n = min(len(clean_type), len(clean_truck), len(bcm_vals),
            len(hour_vals), len(prod_hours), len(distance_vals))
    clean_type   = clean_type.iloc[:n]
    clean_truck  = clean_truck.iloc[:n]
    bcm_vals     = bcm_vals.iloc[:n]
    hour_vals    = hour_vals.iloc[:n]
    load_numeric = load_numeric.iloc[:n]
    prod_hours   = prod_hours.iloc[:n]
    distance_vals= distance_vals.iloc[:n]

    # 6. Filter valid equipment types
    valid_types = ["RDT", "ADT BIG", "RDT SMALL", "ADT"]
    mask = clean_type.isin(valid_types)
    truck = clean_truck[mask]
    bcm   = bcm_vals[mask]
    hours = hour_vals[mask]
    loads = load_numeric.sum(axis=1)[mask]
    prod  = prod_hours[mask]
    dist  = distance_vals[mask]

    # 7. Derived row-level metrics
    row_distance     = 2 * loads * dist
    row_distance_bcm = dist * bcm

    # 8. Aggregate per truck
    monthly = pd.DataFrame({
        'Truck':           truck,
        'TotalBCM':        bcm,
        'TotalHours':      hours,
        'ProductionHours': prod,
        'TotalLoads':      loads,
        'TotalDistance':   row_distance,
        'HaulDistance_x_BCM': row_distance_bcm
    })
    monthly = monthly.groupby('Truck', as_index=False).sum()

    # 9. Add MonthYear and engineered metrics
    monthly['MonthYear']              = month_year
    monthly['BCM_per_EngineHour']     = monthly['TotalBCM'] / monthly['TotalHours']
    monthly['Distance_per_EngineHour']= monthly['TotalDistance'] / monthly['TotalHours']
    monthly['BCM_per_Distance']       = monthly['TotalBCM'] / monthly['TotalDistance']
    monthly['LoadFactor']             = monthly['ProductionHours'] / monthly['TotalHours']
    monthly['AvgLoadVolume']          = monthly['TotalBCM'] / monthly['TotalLoads']
    monthly['AvgHaulDistance']        = monthly['TotalDistance'] / monthly['TotalLoads']

    # 10. Reorder columns
    cols = [
        'MonthYear','Truck',
        'TotalBCM','TotalHours','ProductionHours',
        'TotalLoads','AvgLoadVolume','TotalDistance','AvgHaulDistance',
        'BCM_per_EngineHour','Distance_per_EngineHour','BCM_per_Distance',
        'HaulDistance_x_BCM'
    ]
    all_months.append(monthly[cols])

# === Combine all months and save ===
result = pd.concat(all_months, ignore_index=True)
result.to_excel(output_file, index=False)
print(f"✅ Aggregated {len(files)} files into {output_file}")
