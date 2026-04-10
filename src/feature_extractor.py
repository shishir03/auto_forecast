from pathlib import Path

import xarray as xr
import pandas as pd
from IPython.display import display
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

GFS_BUCKET_NAME = "noaa-gfs-bdp-pds"
MODEL_DIR = "model_files"               # Where to download model files to
FORECAST_MODEL = "atmos"
FILE_TYPE = "pgrb2"
GRID_RESOLUTION = "0p25"

"""
Downloads a file from the GFS AWS S3 bucket.

forecast_date: The model run date in YYYYMMDD format
forecast_cycle: Which model run (00, 06, 12, 18)
forecast_hour: Pretty self-explanatory (in XXX format)
"""
def download_file(forecast_date, forecast_cycle, forecast_hour):
    folder = f"gfs.{forecast_date}/{forecast_cycle}/{FORECAST_MODEL}/"

    try:
        response = s3.list_objects_v2(Bucket=GFS_BUCKET_NAME, Prefix=folder, Delimiter='/')

        # Print common prefixes (subfolders)
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                print(prefix['Prefix'])

        # Print object keys
        if 'Contents' in response and response['Contents']:
            for obj in response['Contents']:
                print(obj['Key'])
        else:
            print("No objects found in the folder.")
    except Exception as e:
        print(f"Error accessing S3: {e}")


    object_key = f"gfs.t{forecast_cycle}z.{FILE_TYPE}.{GRID_RESOLUTION}.f{forecast_hour}"

    remote_object_key = folder + object_key
    local_file_name = f"{MODEL_DIR}/{object_key}"
    s3.download_file(GFS_BUCKET_NAME, remote_object_key, local_file_name)
    print(f"File {object_key} downloaded successfully.")

sample_date = "20250213"
sample_cycle = "00"
sample_hr = "012"

model_filename = f"{MODEL_DIR}/gfs.t{sample_cycle}z.{FILE_TYPE}.{GRID_RESOLUTION}.f{sample_hr}"
model_file = Path(model_filename)
if not model_file.is_file():
    download_file(sample_date, sample_cycle, sample_hr)

ds_500mb = xr.open_dataset(model_filename, engine="cfgrib", filter_by_keys={'typeOfLevel': 'isobaricInhPa', "level": 500, "shortName": "gh"})
# ds_sfc = xr.open_dataset(model_filename, engine="cfgrib", filter_by_keys={'typeOfLevel': 'surface', "stepType": "instant"})
ds_sfc = xr.open_dataset(model_filename, engine="cfgrib", filter_by_keys={"shortName": "prmsl"})

variable_rows = []
for variable in ds_sfc:
    variable_rows.append({
        "Variable": variable,
        "Long Name": ds_sfc[variable].attrs["long_name"],
        "Units": ds_sfc[variable].attrs["units"]
    })

variable_df = pd.DataFrame(variable_rows)
display(variable_df)
