import xarray as xr
import pandas as pd
import numpy as np
from scipy import ndimage
from dask.distributed import Client
from utils import cluster
from utils.config import TrackerConfig

if __name__ == "__main__":

    config = TrackerConfig()
    #config = TrackerConfig().from_yaml("config_mod.yaml")
    
    nc_folder = config.paths.nc_folder
    input_for_clusters = config.files.masked_anomaly_file
    label_cao_output_dir = config.paths.labeled_nc_folder

    anomaly_var = config.dataset.scaled_anomaly_var
    latitude_dim_name = config.dataset.latitude_dim_name
    area_var =  config.dataset.area_var
    label_var = config.dataset.label_var
    label_filtered_var = config.dataset.label_filtered_var

    STDEV_THRESHOLD = config.algorithm.stdev_threshold
    AREA_THRESHOLD = config.algorithm.area_threshold
    DISTANCE_THRESHOLD = config.algorithm.distance_threshold_km

    lon_res = config.dataset.longitude_spatial_res
    lat_res = config.dataset.latitude_spatial_res

    memory_limit = '5GB'  # Limit memory usage to 8GB per worker
    client = Client(n_workers=12,memory_limit=memory_limit)  # Starts a local cluster with memory limits
    print("Running dask at ",client.dashboard_link)

    # Load dataset with chunking (same pattern as your successful code)
    print("Loading dataset...")
    ds = xr.open_dataset("{}{}".format(nc_folder, input_for_clusters))#, 
                         #chunks={'time': 1, 'latitude': 201, 'longitude': 441})
     
    grid_areas = cluster.area_weights(ds[anomaly_var],latitude_dim_name=latitude_dim_name,lon_res=lon_res,lat_res=lat_res)
    
    ds[area_var] = grid_areas.broadcast_like(ds.isel(time=0))

    lon_distance, lat_distance = cluster.calculate_grid_distances(data_array=ds,lon_res=lon_res,lat_res=lat_res)
    pixel_distance = np.sqrt(lon_distance * lat_distance)

    ds['pixel_distance'] = pixel_distance.broadcast_like(ds.isel(time=0))

    #ds['longitude_distance'] = lon_distance.broadcast_like(ds.isel(time=0))
    #ds['latitude_distance'] = lat_distance.broadcast_like(ds.isel(time=0))

    #pixel_distance = calculate_single_pixel_distance(ds, latitude_dim_name=latitude_dim_name)
    #ds['pixel_distance'] = pixel_distance.broadcast_like(ds.isel(time=0))

    ds = ds.persist()  

    def wrapper_for_ufunc(anomaly_block, area_block,distance_block, threshold, area_threshold, distance_threshold, block_id=None):
        results = []
        for i in range(anomaly_block.shape[0]):  
            result = cluster.label_filter_and_merge(anomaly_block[i], area_block, distance_block,threshold, area_threshold,distance_threshold)
            results.append(result)
        return np.stack(results)
    
    # Apply to your data
    filtered_labels = xr.apply_ufunc(
        wrapper_for_ufunc,
        ds[anomaly_var], 
        ds[area_var],
        ds['pixel_distance'],
        STDEV_THRESHOLD,
        AREA_THRESHOLD,
        DISTANCE_THRESHOLD,
        input_core_dims=[['latitude', 'longitude'], ['latitude', 'longitude'], ['latitude', 'longitude'], [], [], []],
        output_core_dims=[['latitude', 'longitude']],
        dask='parallelized',
        output_dtypes=[int],
    )

    # Add processed data to dataset
    ds[label_filtered_var] = filtered_labels
    ds['has_clusters'] = (ds[label_filtered_var] > 0).any(['latitude', 'longitude'])

    years = pd.to_datetime(ds.time.values).year
    unique_years = np.unique(years)

    print(f"Writing {len(unique_years)} years...")
    for year in unique_years:
        print(f"Writing {year}...")
        ds_year = ds.sel(time=str(year))
        ds_year.to_netcdf(f"{label_cao_output_dir}/clusters_{year}.nc", mode="w")
        
    print("Done! Files saved as: clusters_YYYY.nc")


    print("Done!")
    client.close()