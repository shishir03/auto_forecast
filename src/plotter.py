# Plotting code here for testing / debugging purposes
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_contour_field(ds, var_name=None, lows=None, highs=None, title="", cmap='RdYlBu_r'):
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

    if lows is not None:
        for low in lows:
            ax.plot(low['lon'], low['lat'], 'rv',  # red triangle
                    markersize=8, transform=ccrs.PlateCarree())
            ax.text(low['lon'], low['lat'] - 1.5, f"L\n{low['mslp']:.0f}",
                    color='red', fontsize=8, fontweight='bold',
                    ha='center', transform=ccrs.PlateCarree())

    if highs is not None:
        for high in highs:
            ax.plot(high['lon'], high['lat'], 'b^',  # blue triangle
                    markersize=8, transform=ccrs.PlateCarree())
            ax.text(high['lon'], high['lat'] - 1.5, f"H\n{high['mslp']:.0f}",
                    color='blue', fontsize=8, fontweight='bold',
                    ha='center', transform=ccrs.PlateCarree())

    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.05, label=var_name)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_z500_laplacian(ds_z500, z500_smoothed, laplacian):
    _, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-95)}
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)

    lap_scaled = laplacian * 1e5

    vmax = np.percentile(np.abs(lap_scaled), 95)  # robust color scale

    cf = ax.contourf(
        ds_z500['longitude'], ds_z500['latitude'], lap_scaled,
        levels=np.linspace(-vmax, vmax, 21),
        cmap='RdBu_r',
        transform=ccrs.PlateCarree(),
        extend='both'
    )

    z_vals = z500_smoothed.metpy.dequantify()
    cs = ax.contour(
        ds_z500['longitude'], ds_z500['latitude'], z_vals,
        levels=np.arange(4800, 6000, 60),
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

def plot_wind_vectors(wind_speeds, lats, lons, vectors):
    _, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-95)}
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)

    # Plot wind speed as background
    cf = ax.contourf(lons, lats, wind_speeds,
                    levels=np.arange(30, 90, 5),
                    cmap='YlOrRd', transform=ccrs.PlateCarree())
    plt.colorbar(cf, ax=ax, label='Wind Speed (m/s)')

    # Plot vectors
    vector_lats = np.array([v['lat'] for v in vectors])
    vector_lons = np.array([v['lon'] for v in vectors])
    vector_u = np.array([v['u'] for v in vectors])
    vector_v = np.array([v['v'] for v in vectors])

    ax.quiver(vector_lons, vector_lats, vector_u, vector_v,
            transform=ccrs.PlateCarree(),
            scale=1500, width=0.003, color='black')

    ax.set_title('250mb Jet Stream Wind Vectors')
    plt.tight_layout()
    plt.show()
