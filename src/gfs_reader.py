from pathlib import Path

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
        return

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
            print("No objects found in the folder.")
    except Exception as e:
        print(f"Error accessing S3: {e}")

    remote_object_key = folder + object_key
    s3.download_file(GFS_BUCKET_NAME, remote_object_key, local_file_name)
    print(f"File {object_key} downloaded successfully.")
