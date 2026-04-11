import xarray as xr
import metpy.calc as mpcalc
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter

import plotter
from gfs_reader import download_file, FILE_TYPE, GRID_RESOLUTION, MODEL_DIR

sample_date = "20250213"
sample_cycle = "00"
sample_hr = "012"

download_file(sample_date, sample_cycle, sample_hr)

# Try to encompass a good portion of the North Pacific / western North America
lat_min, lat_max = 10, 60
lon_min, lon_max = 180, 260

def open_xr(filename, filter):
    return xr.open_dataset(filename, engine="cfgrib", filter_by_keys=filter) \
            .sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

model_filename = f"{MODEL_DIR}/gfs.t{sample_cycle}z.{FILE_TYPE}.{GRID_RESOLUTION}.f{sample_hr}"

ds_z500 = open_xr(model_filename, {'typeOfLevel': 'isobaricInhPa', "level": 500, "shortName": "gh"})
ds_sfc = open_xr(model_filename, {"typeOfLevel": "surface", "shortName": "t"})
ds_mslp = open_xr(model_filename, {"shortName": "prmsl"})

# plotter.plot_contour_field(ds_mslp, var_name="prmsl", title="MSLP", cmap="RdBu_r")

z500_climo = xr.open_dataset(f"{MODEL_DIR}/hgt.mon.ltm.1991-2020.nc").sel(
    lat=slice(lat_max, lat_min), 
    lon=slice(lon_min, lon_max),
    level=500.0,
    time="0001-02-01 00:00:00"
).squeeze("time").interp(lat=ds_z500['latitude'], lon=ds_z500['longitude'])

# 500 mb height anomalies
z500_anom = ds_z500["gh"] - z500_climo["hgt"]
# plotter.plot_500mb_field(z500_anom, title='500mb Geopotential Height')

# plotter.plot_500mb_field(ds_z500, 'gh', title='500mb Geopotential Height')

def get_z500_laplacian(ds_z500):
    ds_z500_crs = ds_z500.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)
    z500 = ds_z500_crs['gh'].metpy.quantify()
    z500 = xr.apply_ufunc(gaussian_filter, z500, kwargs={'sigma': 5}, dask='parallelized').metpy.quantify()
    laplacian = mpcalc.laplacian(z500, coordinates=(ds_z500_crs['latitude'], ds_z500_crs['longitude']))
    plotter.plot_z500_laplacian(ds_z500, z500, laplacian)
    return laplacian

def get_sfc_features(ds_mslp, neighborhood_size=10, min_depth=2.0):
    lats = ds_mslp["latitude"].values
    lons = ds_mslp["longitude"].values

    mslp_smooth = xr.apply_ufunc(gaussian_filter, ds_mslp["prmsl"], kwargs={'sigma': 3}, dask='parallelized')
    mslp_smooth = mslp_smooth.values
    lows = minimum_filter(mslp_smooth, size=neighborhood_size)
    highs = maximum_filter(mslp_smooth, size=neighborhood_size)
    
    lows_mask = (
        (mslp_smooth == lows) &
        (mslp_smooth < 1013) &
        (mslp_smooth < highs - min_depth)
    )
    highs_mask = (
        (mslp_smooth == highs) &
        (mslp_smooth > 1013) &
        (mslp_smooth > lows + min_depth)
    )

    low_indices = np.argwhere(lows_mask)
    high_indices = np.argwhere(highs_mask)

    lows = [{'lat': float(lats[i]), 'lon': float(lons[j]), 'mslp': float(mslp_smooth[i, j])}
        for i, j in low_indices
    ]

    highs = [{'lat': float(lats[i]), 'lon': float(lons[j]), 'mslp': float(mslp_smooth[i, j])}
        for i, j in high_indices
    ]

    return lows, highs

print(get_sfc_features(ds_mslp))
