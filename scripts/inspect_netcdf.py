import sys
import xarray as xr

if len(sys.argv) < 2:
    print("Usage: python scripts/inspect_netcdf.py <netcdf_file>")
    sys.exit(1)

path = sys.argv[1]

print(f"Opening file: {path}")

try:
    ds = xr.open_dataset(path)
except Exception as e:
    print(f"Failed to open file: {e}")
    sys.exit(1)

print("\n=== VARIABLES ===")
for var in ds.data_vars:
    print(var)

print("\n=== DATASET SUMMARY ===")
print(ds)