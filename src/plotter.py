# Plotting code here for testing / debugging purposes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_contour_field(ds, var_name=None, title="", cmap='RdYlBu_r'):
    _, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-95)}
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3)

    if var_name is not None:
        data = ds[var_name]
    else:
        data = ds

    try:
        lats = ds['latitude'].values
        lons = ds['longitude'].values
    except KeyError:
        lats = ds["lat"].values
        lons = ds["lon"].values

    cf = ax.contourf(lons, lats, data, levels=20,
                     transform=ccrs.PlateCarree(), cmap=cmap)
    cs = ax.contour(lons, lats, data, levels=20,
                    transform=ccrs.PlateCarree(), colors='black', linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%d')

    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, label=var_name)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_z500_laplacian(ds_z500, z500_smoothed, laplacian):
    _, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-95)}
    )

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)

    # Filled contours for the Laplacian
    # The values will be very small (units of m/degree^2 or similar)
    # so you may need to scale - multiply by 1e5 or so to get readable numbers
    lap_scaled = laplacian * 1e5

    vmax = np.percentile(np.abs(lap_scaled), 95)  # robust color scale

    cf = ax.contourf(
        ds_z500['longitude'], ds_z500['latitude'], lap_scaled,
        levels=np.linspace(-vmax, vmax, 21),
        cmap='RdBu_r',
        transform=ccrs.PlateCarree(),
        extend='both'
    )

    # Overlay the raw 500mb height contours for reference - very useful
    # for visually confirming troughs/ridges line up with the Laplacian
    z_vals = z500_smoothed.metpy.dequantify()
    cs = ax.contour(
        ds_z500['longitude'], ds_z500['latitude'], z_vals,
        levels=np.arange(4800, 6000, 60),  # adjust range to your data
        colors='black',
        linewidths=0.8,
        transform=ccrs.PlateCarree()
    )
    ax.clabel(cs, fontsize=8, fmt='%d')

    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05,
                label='500mb Height Laplacian (×10⁻⁵)')
    ax.set_title('500mb Geopotential Height and Laplacian', fontsize=14)

    plt.tight_layout()
    plt.show()
