import pandas as pd
import re
from datetime import datetime
import sys
from pathlib import Path

def col_idx_to_letter(idx: int) -> str:
    """Convert zero-based column index to Excel letter (0 -> A, 25 -> Z, 26 -> AA, etc.)."""
    letters = ''
    while idx >= 0:
        letters = chr(idx % 26 + ord('A')) + letters
        idx = idx // 26 - 1
    return letters

# === File paths ===
agg_file    = 'Correct Totals.xlsx'  # aggregated output
diesel_file = 'Diesel.xlsx'
output_file = 'Correct Fuel.xlsx'

# === Read aggregated metrics ===
agg_df = pd.read_excel(agg_file)
if 'MonthYear' not in agg_df.columns or 'Truck' not in agg_df.columns:
    print('Error: aggregated file must contain MonthYear and Truck columns.')
    sys.exit(1)

# === Read Diesel data without headers ===
diesel_raw = pd.read_excel(diesel_file, header=None)

# === Build mapping of date -> diesel column index ===
start_col = 21  # Excel V
end_col   = 80  # Excel CC

date_to_col = {}
for col_idx in range(start_col, min(end_col+1, diesel_raw.shape[1])):
    cell = diesel_raw.iat[2, col_idx]  # row 3 -> index 2
    if pd.isna(cell):
        continue
    try:
        if isinstance(cell, datetime):
            cell_date = cell.date()
        else:
            cell_date = pd.to_datetime(cell).date()
    except Exception:
        continue
    date_to_col[cell_date] = col_idx

# === Helper to parse MonthYear from aggregated file ===
def parse_monthyear(x):
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x.date()
    s = str(x).strip()
    for fmt in ('%B %y','%b %y','%b%y'):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            pass
    try:
        return pd.to_datetime(s).date()
    except:
        return None

# === Extract DieselTotal and SourceCell for each aggregated row ===
fuel_values  = []
source_cells = []

for _, row in agg_df.iterrows():
    # 1. Column by date
    date_input = parse_monthyear(row['MonthYear'])
    col_idx    = date_to_col.get(date_input)
    if col_idx is None:
        fuel_values.append(None)
        source_cells.append(None)
        continue

    # 2. Extract code from Truck
    truck = str(row['Truck']).upper()
    m = re.search(r"(\d{3}).*-(\d{3})", truck)
    if not m:
        fuel_values.append(None)
        source_cells.append(None)
        continue
    code = f"{m.group(1)}-{m.group(2)}"

    # 3. Find matching row in Diesel column R (index 17)
    equip_col_idx = 17
    match_row     = None
    for ridx, cell in enumerate(diesel_raw.iloc[:, equip_col_idx]):
        if pd.isna(cell):
            continue
        if code in str(cell).upper():
            match_row = ridx
            break

    if match_row is None:
        fuel_values.append(None)
        source_cells.append(None)
    else:
        fuel_values.append(diesel_raw.iat[match_row, col_idx])
        # build Excel cell address: column letter + (row index + 1)
        cell_addr = f"{col_idx_to_letter(col_idx)}{match_row+1}"
        source_cells.append(cell_addr)

# === Append and save ===
agg_df['DieselTotal'] = fuel_values
agg_df['SourceCell']  = source_cells
agg_df.to_excel(output_file, index=False)
print(f"✅ Merged DieselTotal with source cell into {output_file}")
