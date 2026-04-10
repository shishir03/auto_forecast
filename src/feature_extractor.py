import xarray as xr
import metpy.calc as mpcalc
from scipy.ndimage import gaussian_filter

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

z500_climo = xr.open_dataset(f"{MODEL_DIR}/hgt.mon.ltm.1991-2020.nc").sel(
    lat=slice(lat_max, lat_min), 
    lon=slice(lon_min, lon_max),
    level=500.0,
    time="0001-02-01 00:00:00"
).squeeze("time").interp(lat=ds_z500['latitude'], lon=ds_z500['longitude'])

z500_anom = ds_z500["gh"] - z500_climo["hgt"]
plotter.plot_500mb_field(z500_anom, title='500mb Geopotential Height')

# plotter.plot_500mb_field(ds_z500, 'gh', title='500mb Geopotential Height')

'''
ds_z500 = ds_z500.metpy.assign_crs(grid_mapping_name='latitude_longitude', earth_radius=6371229.0)
z500 = ds_z500['gh'].metpy.quantify()
z500 = xr.apply_ufunc(gaussian_filter, z500, kwargs={'sigma': 3}, dask='parallelized').metpy.quantify()
laplacian = mpcalc.laplacian(z500, coordinates=(ds_z500['latitude'], ds_z500['longitude'])) '''

# plotter.plot_z500_laplacian(ds_z500, z500, laplacian)
