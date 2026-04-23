from pathlib import Path
import xarray as xr

import boto3
from botocore import UNSIGNED
from botocore.config import Config

GFS_BUCKET_NAME = "noaa-gfs-bdp-pds"
MODEL_DIR = "model_files"               # Where to download model files to
FORECAST_MODEL = "atmos"
FILE_TYPE = "pgrb2"
GRID_RESOLUTION = "0p25"

def download_file(forecast_date, forecast_cycle, forecast_hour, verbose=False):
    """
    Downloads a file from the GFS AWS S3 bucket.

    forecast_date: The model run date in YYYYMMDD format
    forecast_cycle: Which model run (00, 06, 12, 18)
    forecast_hour: Pretty self-explanatory (in XXX format)
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    object_key = f"gfs.t{forecast_cycle}z.{FILE_TYPE}.{GRID_RESOLUTION}.f{forecast_hour}"
    local_file_name = f"{MODEL_DIR}/{forecast_date}{forecast_cycle}{forecast_hour}.{object_key}"
    model_file = Path(local_file_name)
    if model_file.is_file():
        print("File already found")
        return local_file_name

    model_file.parent.mkdir(exist_ok=True, parents=True)
    
    folder = f"gfs.{forecast_date}/{forecast_cycle}/{FORECAST_MODEL}/"

    try:
        response = s3.list_objects_v2(Bucket=GFS_BUCKET_NAME, Prefix=folder, Delimiter='/')

        # Print common prefixes (subfolders)
        if 'CommonPrefixes' in response:
            if verbose:
                for prefix in response['CommonPrefixes']:
                    print(prefix['Prefix'])

        # Print object keys
        if 'Contents' in response and response['Contents']:
            if verbose:
                for obj in response['Contents']:
                    print(obj['Key'])
        else:
            print(f"No objects found in folder {folder}")
    except Exception as e:
        print(f"Error accessing S3: {e}")

    remote_object_key = folder + object_key
    s3.download_file(GFS_BUCKET_NAME, remote_object_key, local_file_name)
    print(f"File {object_key} downloaded successfully.")
    return local_file_name

sample_date = "20260217"
sample_cycle = "00"
sample_hr = "012"

# download_file(sample_date, sample_cycle, sample_hr)
model_filename = f"{MODEL_DIR}/{sample_date}{sample_cycle}{sample_hr}.gfs.t{sample_cycle}z.{FILE_TYPE}.{GRID_RESOLUTION}.f{sample_hr}"

"""
Grid sizes are as follows:

1. Big grid (for synoptic-scale patterns)
2. Medium grid (smaller but still synoptic-scale, for coarse-resolution wind data / PWATs)
3. Small grid (for hi-res observations like temperature / precipitation)
"""
grid_sizes = [(10, 60, 180, 260), (25, 50, 225, 255), (36, 38.5, 236, 239)]

def open_xr(filter, filename=model_filename, grid=0):
    lat_min, lat_max, lon_min, lon_max = grid_sizes[grid]
    return xr.open_dataset(filename, engine="cfgrib", filter_by_keys=filter, decode_timedelta=True) \
        .sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

def read_grids(filename=model_filename):
    # Big grids
    ds_z500 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "gh"}, filename=filename)
    ds_mslp = open_xr({"shortName": "prmsl"}, filename=filename) / 100
    ds_u250 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 250, "shortName": "u"}, filename=filename)
    ds_v250 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 250, "shortName": "v"}, filename=filename)

    # Medium grids
    ds_u500 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "u"}, filename=filename, grid=1)
    ds_v500 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "v"}, filename=filename, grid=1)
    ds_u850 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 850, "shortName": "u"}, filename=filename, grid=1)
    ds_v850 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 850, "shortName": "v"}, filename=filename, grid=1)
    ds_usfc = open_xr({"shortName": "10u"}, filename=filename, grid=1)
    ds_vsfc = open_xr({"shortName": "10v"}, filename=filename, grid=1)
    ds_pwat = open_xr({"shortName": "pwat"}, filename=filename, grid=1)

    # Small grids
    ds_t500 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "t"}, filename=filename, grid=2)
    ds_t850 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 850, "shortName": "t"}, filename=filename, grid=2)
    ds_tsfc = open_xr({"typeOfLevel": "surface", "shortName": "t"}, filename=filename, grid=2)
    ds_cwat = open_xr({"shortName": "cwat"}, filename=filename, grid=2)
    ds_prate = open_xr({"shortName": "prate", "stepType": "avg"}, filename=filename, grid=2)

    lat_min, lat_max, lon_min, lon_max = grid_sizes[0]
    z500_climo = xr.open_dataset(f"{MODEL_DIR}/hgt.mon.ltm.1991-2020.nc", use_cftime=True).sel(
        lat=slice(lat_max, lat_min), 
        lon=slice(lon_min, lon_max),
        level=500.0,
        time=f"0001-{sample_date[4:6]}-01 00:00:00"         # Get climatology for the right month
    ).squeeze("time").interp(lat=ds_z500["latitude"], lon=ds_z500["longitude"])

    # 500 mb height anomalies
    z500_anom = ds_z500["gh"] - z500_climo["hgt"]

    return ds_mslp, z500_anom, ds_u250, ds_v250
