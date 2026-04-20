from bisect import bisect_left
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter

import plotter
from gfs_reader import download_file, FILE_TYPE, GRID_RESOLUTION, MODEL_DIR

sample_date = "20260217"
sample_cycle = "00"
sample_hr = "012"

download_file(sample_date, sample_cycle, sample_hr)
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
    return xr.open_dataset(filename, engine="cfgrib", filter_by_keys=filter) \
        .sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))

# Big grids
ds_z500 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "gh"})
ds_mslp = open_xr({"shortName": "prmsl"}) / 100
ds_u250 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 250, "shortName": "u"})
ds_v250 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 250, "shortName": "v"})

# Medium grids
ds_u500 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "u"}, grid=1)
ds_v500 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "v"}, grid=1)
ds_u850 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 850, "shortName": "u"}, grid=1)
ds_v850 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 850, "shortName": "v"}, grid=1)
ds_usfc = open_xr({"shortName": "10u"}, grid=1)
ds_vsfc = open_xr({"shortName": "10v"}, grid=1)
ds_pwat = open_xr({"shortName": "pwat"}, grid=1)

# Small grids
ds_t500 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "t"}, grid=2)
ds_t850 = open_xr({"typeOfLevel": "isobaricInhPa", "level": 850, "shortName": "t"}, grid=2)
ds_tsfc = open_xr({"typeOfLevel": "surface", "shortName": "t"}, grid=2)
ds_cwat = open_xr({"shortName": "cwat"}, grid=2)
ds_prate = open_xr({"shortName": "prate", "stepType": "avg"}, grid=2)

lat_min, lat_max, lon_min, lon_max = grid_sizes[0]
z500_climo = xr.open_dataset(f"{MODEL_DIR}/hgt.mon.ltm.1991-2020.nc").sel(
    lat=slice(lat_max, lat_min), 
    lon=slice(lon_min, lon_max),
    level=500.0,
    time=f"0001-{sample_date[4:6]}-01 00:00:00"         # Get climatology for the right month
).squeeze("time").interp(lat=ds_z500["latitude"], lon=ds_z500["longitude"])

# 500 mb height anomalies
z500_anom = ds_z500["gh"] - z500_climo["hgt"]
# plotter.plot_500mb_field(z500_anom, title="500mb Geopotential Height")

# plotter.plot_contour_field(ds_z500, var_name="gh", title="500mb Geopotential Height")

def get_z500_laplacian(ds_z500):
    ds_z500_crs = ds_z500.metpy.assign_crs(grid_mapping_name="latitude_longitude", earth_radius=6371229.0)
    z500 = ds_z500_crs["gh"].metpy.quantify()
    z500 = xr.apply_ufunc(gaussian_filter, z500, kwargs={"sigma": 5}, dask="parallelized").metpy.quantify()
    laplacian = mpcalc.laplacian(z500, coordinates=(ds_z500_crs["latitude"], ds_z500_crs["longitude"]))
    plotter.plot_z500_laplacian(ds_z500, z500, laplacian)
    return z500, laplacian

def get_sfc_features(ds_mslp, neighborhood_size=10, min_depth=2.0):
    """
    Determine high or low pressure centers from an MSLP field
    """
    lats = ds_mslp["latitude"].values
    lons = ds_mslp["longitude"].values

    mslp_smooth = xr.apply_ufunc(gaussian_filter, ds_mslp["prmsl"], kwargs={"sigma": 3}, dask="parallelized")
    mslp_smooth = mslp_smooth.values
    lows = minimum_filter(mslp_smooth, size=neighborhood_size)
    highs = maximum_filter(mslp_smooth, size=neighborhood_size)
    
    lows_mask = ((mslp_smooth == lows) & (mslp_smooth < 1013) & (mslp_smooth < highs - min_depth))
    highs_mask = ((mslp_smooth == highs) & (mslp_smooth > 1013) & (mslp_smooth > lows + min_depth))

    low_indices = np.argwhere(lows_mask)
    high_indices = np.argwhere(highs_mask)

    lows = [{"lat": float(lats[i]), "lon": float(lons[j]), "mslp": float(mslp_smooth[i, j])}
        for i, j in low_indices
    ]

    highs = [{"lat": float(lats[i]), "lon": float(lons[j]), "mslp": float(mslp_smooth[i, j])}
        for i, j in high_indices
    ]

    return lows, highs

def get_wind_vectors(ds_u, ds_v, jet_threshold=30, spacing_deg=5.0, neighborhood_size=10, local_maxima=False):
    """
    Get spaced out wind vectors within the "core" of the jet (250mb winds above the given threshold)
    """
    u_vals = ds_u["u"].values
    v_vals = ds_v["v"].values
    ds_wind250 = mpcalc.wind_speed(u_vals * units("m/s"), v_vals * units("m/s"))
    # ds_wind250 = xr.apply_ufunc(gaussian_filter, ds_wind250, kwargs={"sigma": 3}, dask="parallelized")

    lats = ds_u["latitude"].values
    lons = ds_u["longitude"].values

    if local_maxima:
        mask = ds_wind250 >= jet_threshold
        jet_masked = np.where(mask, ds_wind250, 0)
        local_max = maximum_filter(jet_masked, size=neighborhood_size)
        points = np.argwhere((jet_masked == local_max) & mask)
    else:
        dlat = abs(lats[1] - lats[0])
        dlon = abs(lons[1] - lons[0])
        lat_stride = max(1, int(spacing_deg / dlat))
        lon_stride = max(1, int(spacing_deg / dlon))

        points = [np.arange(0, len(lats), lat_stride), np.arange(0, len(lons), lon_stride)]
        np.reshape(points, (len(lats), 2))

    vectors = []

    for p in points:
        i = p[0]
        j = p[1]

        if ds_wind250[i, j] < jet_threshold * units("m/s"):
            continue

        wspd = float(ds_wind250[i, j].magnitude)
        u = float(u_vals[i, j])
        v = float(v_vals[i, j])

        wind_dir = float(mpcalc.wind_direction(u * units("m/s"), v * units("m/s")).magnitude)
        directions = np.arange(22.5, 360, 45)
        labels = ["NE", "E", "SE", "S", "SW", "W", "NW", "N"]
        flow_label = labels[bisect_left(directions, wind_dir) - 1]

        vectors.append({
            "lat": float(lats[i]),
            "lon": float(lons[j]),
            "u": u,
            "v": v,
            "wspd": wspd,
            "direction": wind_dir,
            "flow_label": flow_label
        })

    return vectors

lows, highs = get_sfc_features(ds_mslp)
plotter.plot_contour_field(ds_mslp, var_name="prmsl", lows=lows, highs=highs, title="MSLP Plot", cmap="RdBu_r")

# ds_wind250 = mpcalc.wind_speed(ds_u250["u"].metpy.quantify(), ds_v250["v"].metpy.quantify())
# vectors = get_wind_vectors(ds_u250, ds_v250, spacing_deg=2.5)
# plotter.plot_wind_vectors(ds_wind250, ds_u250["latitude"].values, ds_v250["longitude"].values, vectors)
def features_to_text(sfc_features, tempers, tempers_850mb, vectors_250mb, vectors_850mb, z500_anom):
    lines = []
    
    lows, highs = sfc_features
    def output_features(features, low_or_high):
        for f in features:
            north_south = "N" if f["lat"] >= 0 else "S"
            adjusted_lon = f["lon"] % 180 - 180 * (f["lon"] // 180)
            east_west = "E" if adjusted_lon >= 0 else "W"
            lines.append(f"Surface {low_or_high} at {(abs(f["lat"])):.1f}{north_south}, {(abs(adjusted_lon)):.1f}{east_west}, "
                        f"central pressure {f["mslp"]:.0f} hPa")
    
    output_features(lows, "low")
    output_features(highs, "high")

    if not vectors_250mb or len(vectors_250mb) == 0:
        lines.append("No significant jet stream activity")
    else:
        for i, v in enumerate(vectors_250mb):
            lines.append(
                f"Jet streak {i+1}: {v["wspd"]:.0f} m/s {v["flow_label"]} flow "
                f"at {v["lat"]:.1f}N, {v["lon"]:.1f}E"
            )
    
    return "\n".join(lines)
