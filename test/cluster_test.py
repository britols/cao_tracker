import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage
import os

from utils import cluster

from utils.config import(
    data_path,
    test_path,
    csv_folder,
    STDEV_THRESHOLD,
    AREA_THRESHOLD,
    input_for_clusters,
    output_from_clusters,
    output_from_clusters_table,
    anomaly_dim,
    latitude_dim_name,
    mask_dim,
    area_dim,
    label_dim,
    time_dim,
    label_filtered_dim
)

#Open input file
ds = xr.open_dataset("{}{}".format(data_path,input_for_clusters))
#Mask where stdev < threshold
ds[mask_dim] = ds[anomaly_dim] <= STDEV_THRESHOLD

#Calculate and broadcast area of each grid cell
grid_areas = cluster.area_weights(ds[mask_dim],
                                  latitude_dim_name=latitude_dim_name)
ds[area_dim] = grid_areas.broadcast_like(ds.isel(time=1))

#Label clusters and filter by area size
ds=ds.groupby(time_dim).map(cluster.label_and_filter,
                             mask_dim=mask_dim,
                             area_dim=area_dim,
                            label_dim=label_dim,
                            label_filtered_dim=label_filtered_dim,
                             AREA_THRESHOLD=AREA_THRESHOLD)
#Get information about filtered clusters
list_df = list()
for t in ds.where(ds.has_clusters,drop=True).time.values:
    data_set = ds.sel(time=t)
    info_temp = cluster.get_cluster_info(data_set,
                                            label_dim=label_dim,
                                            label_filtered_dim=label_filtered_dim,
                                            anomaly_dim=anomaly_dim,
                                            area_dim=area_dim)
    list_df.append(info_temp)

cluster_info_df = pd.concat(list_df,axis=0)

#Save filtered clusters to file
ds[label_filtered_dim].to_netcdf(os.path.join('{}{}'.format(data_path,output_from_clusters)),compute=True,mode="w")
cluster_info_df.to_csv(os.path.join('{}{}{}'.format(test_path,csv_folder,output_from_clusters_table)),sep=";",index=False)
