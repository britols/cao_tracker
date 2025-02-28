import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import pandas as pd
import os

from utils import plot_utils

from utils.config import(
    data_path,
    test_path,
    csv_folder,
    image_folder,
    input_for_clusters,
    output_from_clusters,
    output_from_clusters_table,
    anomaly_dim,
    mask_dim,
    label_dim,
    time_dim,
    label_filtered_dim
)


new_cmap_viridis = plot_utils.set_white_color(plt.cm.viridis)
new_cmap_set1 = plot_utils.set_white_color(plt.cm.Set1)

ds_clusters = xr.open_dataset("{}{}".format(data_path,output_from_clusters))
ds_anomalies = xr.open_dataset("{}{}".format(data_path,input_for_clusters))
df_clusters_info = pd.read_csv("{}{}{}".format(test_path,csv_folder,output_from_clusters_table),sep=";")

t = '1985-02-07'
da_clusters = ds_clusters.sel(time=t)[label_filtered_dim]
da_anomalies = ds_anomalies.sel(time=t)[anomaly_dim]

#==== test_plot_01.png=====================================================
plt.figure(figsize=(12, 6))
plot_utils.map_plot()
plot_utils.da_plot(da_clusters,cmap=new_cmap_viridis)
plt.savefig("{}{}test_plot_01.png".format(test_path,image_folder))
#===========================================================================

#==== test_plot_02.png=====================================================
plt.figure(figsize=(12, 6))
plot_utils.map_plot()
plot_utils.da_plot_zero_centered(da_anomalies)
plot_utils.da_plot(da_clusters,cmap=new_cmap_viridis,type="contour")
plt.savefig("{}{}test_plot_02.png".format(test_path,image_folder))
#===========================================================================


#==== test_plot_03.png=====================================================
new_coords = plot_utils.convert_cmass(cm_lat=df_clusters_info["cm_lat"],cm_lon=df_clusters_info["cm_lon"],latitude=ds_clusters.latitude.values,longitude=ds_clusters.longitude.values)
new_lons = list()
new_lats = list()
for c in new_coords:
    new_lons.append(c[0])
    new_lats.append(c[1])

df_clusters_info["cm_lat"] = new_lats
df_clusters_info["cm_lon"] = new_lons
new_cmap_viridis = plot_utils.set_white_color(plt.cm.viridis)
cm_lat=df_clusters_info[df_clusters_info.time=='1985-02-07']['cm_lat']
cm_lon=df_clusters_info[df_clusters_info.time=='1985-02-07']['cm_lon']


plt.figure(figsize=(12, 6))
plot_utils.map_plot()
plot_utils.da_plot(da_clusters,cmap=new_cmap_viridis)
plt.scatter(cm_lon,cm_lat)
plt.savefig("{}{}test_plot_03.png".format(test_path,image_folder))
#==== 