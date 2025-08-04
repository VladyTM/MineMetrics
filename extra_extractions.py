import pandas as pd
import re
from datetime import datetime
import sys
from pathlib import Path

# === Helper functions ===
def col_idx_to_letter(idx: int) -> str:
    """Convert zero-based column index to Excel column letter."""
    letters = ''
    while idx >= 0:
        letters = chr(idx % 26 + ord('A')) + letters
        idx = idx // 26 - 1
    return letters

# === Parse MonthYear helper ===
def parse_monthyear(x):
    if pd.isna(x):
        return None
    if isinstance(x, datetime):
        return x.date()
    s = str(x).strip()
    for fmt in ('%B %y', '%b %y', '%b%y'):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    try:
        return pd.to_datetime(s).date()
    except:
        return None

# === Configuration ===
agg_file    = Path('Correct Totals.xlsx')  # aggregated results
diesel_file = Path('Diesel.xlsx')
output_file = Path('Correct Fuel Multi.xlsx')

# === Validate files ===
if not agg_file.exists() or not diesel_file.exists():
    print(f"Error: ensure '{agg_file.name}' and '{diesel_file.name}' exist.")
    sys.exit(1)

# === Read data ===
agg_df    = pd.read_excel(agg_file)
diesel_raw = pd.read_excel(diesel_file, header=None)

# Check required columns
if 'MonthYear' not in agg_df.columns or 'Truck' not in agg_df.columns:
    print("Error: aggregated file must contain MonthYear and Truck columns.")
    sys.exit(1)

# === Define metric column ranges ===
metric_ranges = {
    'DieselTotal': (21, 80),      # V–CC
    'BDHours':      (272, 331),    # JM–LT
    'SBHours':      (335, 394),    # LX–OE
    'LPerEngineHr': (398, 457),    # OI–QP
    'BCM_per_EH':   (524, 583),    # TE–VL
}

# === Build date->col mappings for each metric ===
date_to_col = {m: {} for m in metric_ranges}
for metric, (start, end) in metric_ranges.items():
    for col in range(start, min(end+1, diesel_raw.shape[1])):
        header = diesel_raw.iat[2, col]  # row3 index2
        if pd.isna(header):
            continue
        try:
            if isinstance(header, datetime):
                hdate = header.date()
            else:
                hdate = pd.to_datetime(header).date()
        except:
            continue
        date_to_col[metric][hdate] = col

# === Extract metrics per row ===
fuel_vals      = []
bdh_vals       = []
sbh_vals       = []
lper_eh_vals   = []
bcm_per_eh_vals= []

for _, row in agg_df.iterrows():
    date_input = parse_monthyear(row['MonthYear'])
    # find row index for Truck in Diesel sheet
    truck_code = str(row['Truck']).upper()
    m = re.search(r"(\d{3}).*-(\d{3})", truck_code)
    match_row = None
    if m:
        code = f"{m.group(1)}-{m.group(2)}"
        for ridx, cell in enumerate(diesel_raw.iloc[:, 17]):  # column R idx17
            if pd.isna(cell):
                continue
            if code in str(cell).upper():
                match_row = ridx
                break
    # helper to get metric
    def get_metric(metric):
        col_map = date_to_col.get(metric, {})
        col_idx = col_map.get(date_input)
        if match_row is None or col_idx is None:
            return None
        return diesel_raw.iat[match_row, col_idx]
    # append
    fuel_vals.append(get_metric('DieselTotal'))
    bdh_vals.append(get_metric('BDHours'))
    sbh_vals.append(get_metric('SBHours'))
    lper_eh_vals.append(get_metric('LPerEngineHr'))
    bcm_per_eh_vals.append(get_metric('BCM_per_EH'))

# === Append to DataFrame ===
agg_df['DieselTotal']   = fuel_vals
agg_df['BDHours']        = bdh_vals
agg_df['SBHours']        = sbh_vals
agg_df['LPerEngineHr']   = lper_eh_vals
agg_df['BCM_per_EH']     = bcm_per_eh_vals

# === Save ===
agg_df.to_excel(output_file, index=False)
print(f"✅ Merged multi metrics into {output_file}")
