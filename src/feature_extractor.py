from bisect import bisect_left
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np
from scipy.ndimage import gaussian_filter, minimum_filter, maximum_filter

import plotter
from gfs_reader import read_grids
# plotter.plot_500mb_field(z500_anom, title="500mb Geopotential Height")

# plotter.plot_contour_field(ds_z500, var_name="gh", title="500mb Geopotential Height")

def get_z500_laplacian(ds_z500):
    ds_z500_crs = ds_z500.metpy.assign_crs(grid_mapping_name="latitude_longitude", earth_radius=6371229.0)
    z500 = ds_z500_crs["gh"].metpy.quantify()
    z500 = xr.apply_ufunc(gaussian_filter, z500, kwargs={"sigma": 5}, dask="parallelized").metpy.quantify()
    laplacian = mpcalc.laplacian(z500, coordinates=(ds_z500_crs["latitude"], ds_z500_crs["longitude"]))
    plotter.plot_z500_laplacian(ds_z500, z500, laplacian)
    return z500, laplacian

def get_lows_highs(ds_pressure, field=None, var_name="mslp", neighborhood_size=10, min_depth=2.0, low_bound=1013, high_bound=1013):
    """
    Determine high or low pressure centers from an MSLP field
    """
    lats = ds_pressure["latitude"].values
    lons = ds_pressure["longitude"].values

    if field is not None:
        pressure_field = ds_pressure[field]
    else:
        pressure_field = ds_pressure

    smoothed = xr.apply_ufunc(gaussian_filter, pressure_field, kwargs={"sigma": 3}, dask="parallelized")
    smoothed = smoothed.values
    lows = minimum_filter(smoothed, size=neighborhood_size)
    highs = maximum_filter(smoothed, size=neighborhood_size)
    
    lows_mask = ((smoothed == lows) & (smoothed < low_bound) & (smoothed < highs - min_depth))
    highs_mask = ((smoothed == highs) & (smoothed > high_bound) & (smoothed > lows + min_depth))

    low_indices = np.argwhere(lows_mask)
    high_indices = np.argwhere(highs_mask)

    lows = [{"lat": float(lats[i]), "lon": float(lons[j]), var_name: float(smoothed[i, j])}
        for i, j in low_indices
    ]

    highs = [{"lat": float(lats[i]), "lon": float(lons[j]), var_name: float(smoothed[i, j])}
        for i, j in high_indices
    ]

    return lows, highs

def get_wind_vectors(ds_u, ds_v, jet_threshold=30, spacing_deg=5.0, neighborhood_size=100, local_maxima=False):
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
        mask = ds_wind250 >= (jet_threshold * units("m/s"))
        jet_masked = np.where(mask, ds_wind250.m, 0)
        local_max = maximum_filter(jet_masked, size=neighborhood_size)
        points = np.argwhere((np.abs(jet_masked - local_max) < 1e-6) & mask)
    else:
        dlat = abs(lats[1] - lats[0])
        dlon = abs(lons[1] - lons[0])
        lat_stride = max(1, int(spacing_deg / dlat))
        lon_stride = max(1, int(spacing_deg / dlon))

        points = np.array(np.meshgrid(
            np.arange(0, len(lats), lat_stride), 
            np.arange(0, len(lons), lon_stride)
        )).T.reshape(-1, 2)

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

# lows, highs = get_lows_highs(ds_mslp)
# plotter.plot_contour_field(ds_mslp, var_name="prmsl", lows=lows, highs=highs, title="MSLP Plot", cmap="RdBu_r")

# ds_wind250 = mpcalc.wind_speed(ds_u250["u"].metpy.quantify(), ds_v250["v"].metpy.quantify())
# vectors = get_wind_vectors(ds_u250, ds_v250, spacing_deg=2.5)
# plotter.plot_wind_vectors(ds_wind250, ds_u250["latitude"].values, ds_v250["longitude"].values, vectors)
def features_to_text(ds_mslp, z500_anom, ds_u250, ds_v250):
    lines = []
    
    lows, highs = get_lows_highs(ds_mslp, field="prmsl")
    troughs, ridges = get_lows_highs(z500_anom, var_name="anomaly", neighborhood_size=15, min_depth=20, low_bound=-50, high_bound=50)

    def output_features(features, feature_name, var_name, units):
        for f in features:
            north_south = "N" if f["lat"] >= 0 else "S"
            adjusted_lon = f["lon"] % 180 - 180 * (f["lon"] // 180)
            east_west = "E" if adjusted_lon >= 0 else "W"
            lines.append(f"{feature_name} at {(abs(f["lat"])):.1f}{north_south}, {(abs(adjusted_lon)):.1f}{east_west}, "
                        f"{var_name}: {f[var_name]:.0f} {units}")
    
    # Surface lows / highs
    output_features(lows, "surface low", "mslp", "hPa")
    output_features(highs, "surface high", "mslp", "hPa")

    # Troughs / ridges
    output_features(troughs, "500mb trough", "anomaly", "m")
    output_features(ridges, "500mb ridge", "anomaly", "m")

    # Jet streaks
    vectors_250mb = get_wind_vectors(ds_u250, ds_v250, spacing_deg=2.5, local_maxima=True)
    vectors_250mb.sort(key=lambda x: x["wspd"], reverse=True)

    if not vectors_250mb or len(vectors_250mb) == 0:
        lines.append("No significant jet stream activity")
    else:
        for i, v in enumerate(vectors_250mb[:10]):
            north_south = "N" if v["lat"] >= 0 else "S"
            adjusted_lon = v["lon"] % 180 - 180 * (v["lon"] // 180)
            east_west = "E" if adjusted_lon >= 0 else "W"
            lines.append(
                f"Jet streak {i+1}: {v["wspd"]:.0f} m/s {v["flow_label"]} flow "
                f"at {(abs(v["lat"])):.1f}{north_south}, {(abs(adjusted_lon)):.1f}{east_west}"
            )
    
    return "\n".join(lines)

ds_mslp, z500_anom, ds_u250, ds_v250 = read_grids()
print(features_to_text(ds_mslp, z500_anom, ds_u250, ds_v250))
