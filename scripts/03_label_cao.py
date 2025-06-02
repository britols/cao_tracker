import xarray as xr
import pandas as pd
import numpy as np
from scipy import ndimage
from dask.distributed import Client
from utils import cluster
from utils.config import TrackerConfig


def merge_nearby_labels_2d(labeled_array,distance_array=None,distance_threshold_km=1000,pixel_res_km=None):
    """Merge nearby labels in a 2D array"""
    unique_labels = np.unique(labeled_array)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) <= 1:
        return labeled_array  # Nothing to merge
    
    # Calculate areas for prioritization
    label_areas = {label: np.sum(labeled_array == label) for label in unique_labels}
    
    # Vectorized distance calculation
    regions_stack = np.stack([
        (labeled_array == label) for label in unique_labels
    ], axis=0)
    
    distance_stack = np.stack([
        ndimage.distance_transform_edt(~regions_stack[i])
        for i in range(len(unique_labels))
    ], axis=0)
    
    if distance_array is not None:
        labeled_mask = labeled_array > 0
        pixel_res_km = np.mean(distance_array[labeled_mask]) if np.sum(labeled_mask) > 0 else np.mean(distance_array)

    # Find pairs to merge
    min_distances = {}
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels[i+1:], i+1):
            distances_at_region_j = distance_stack[i][regions_stack[j]]
            min_pixel_dist = distances_at_region_j.min()
            min_km_dist = min_pixel_dist * pixel_res_km
            min_distances[(label1, label2)] = min_km_dist
    
    # Union-Find with area prioritization
    parent = {label: label for label in unique_labels}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    for (label1, label2), distance in min_distances.items():
        if distance < distance_threshold_km:
            root1, root2 = find(label1), find(label2)
            
            # Larger area wins
            if label_areas[root1] >= label_areas[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
    
    # Apply merging
    merged_labels = labeled_array.copy()
    for label in unique_labels:
        root_label = find(label)
        if root_label != label:
            merged_labels[merged_labels == label] = root_label
    
    return merged_labels


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

    memory_limit = '5GB'  # Limit memory usage to 8GB per worker
    client = Client(n_workers=12,memory_limit=memory_limit)  # Starts a local cluster with memory limits
    print("Running dask at ",client.dashboard_link)

    # Load dataset with chunking (same pattern as your successful code)
    print("Loading dataset...")
    ds = xr.open_dataset("{}{}".format(nc_folder, input_for_clusters), 
                         chunks={'time': 1, 'latitude': 201, 'longitude': 441})
     
    grid_areas = cluster.area_weights(ds[anomaly_var],
                                  latitude_dim_name=latitude_dim_name)
    
    ds[area_var] = grid_areas.broadcast_like(ds.isel(time=0))

    lon_distance, lat_distance = cluster.calculate_grid_distances(data_array=ds)
    pixel_distance = np.sqrt(lon_distance * lat_distance)

    ds['pixel_distance'] = pixel_distance.broadcast_like(ds.isel(time=0))

    #ds['longitude_distance'] = lon_distance.broadcast_like(ds.isel(time=0))
    #ds['latitude_distance'] = lat_distance.broadcast_like(ds.isel(time=0))

    #pixel_distance = calculate_single_pixel_distance(ds, latitude_dim_name=latitude_dim_name)
    #ds['pixel_distance'] = pixel_distance.broadcast_like(ds.isel(time=0))

    ds = ds.persist()  

    def process_single_day(anomaly_data, area_data,  distance_data=None,threshold=STDEV_THRESHOLD, area_threshold=AREA_THRESHOLD):
        """Process one day: mask -> label -> filter by area"""
        
        # Step 1: Create mask
        mask = anomaly_data <= threshold
        
        # Step 2: Label connected regions
        labeled_array, _ = ndimage.label(mask)
        
        # Step 3: Filter by area
        if labeled_array.max() == 0:  # No regions found
            return np.zeros_like(labeled_array)
        
        # Calculate areas for each label
        unique_labels = np.unique(labeled_array)
        areas = ndimage.sum_labels(area_data, labeled_array, unique_labels)
        
        # Keep only large enough regions
        valid_labels = unique_labels[(areas > area_threshold) & (unique_labels > 0)]
        
        # Create filtered result
        filtered_result = np.where(np.isin(labeled_array, valid_labels), labeled_array, 0)
        
        if(len(valid_labels)>=2):
            filtered_result = merge_nearby_labels_2d(filtered_result, distance_array=distance_data)

        return filtered_result

    def wrapper_for_map_blocks(anomaly_block, area_block,distance_block, block_id=None):
        #print(f"Block shape: anomaly={anomaly_block.shape}, area={area_block.shape}")
        #print(f"Block ranges: anomaly=({anomaly_block.min():.3f}, {anomaly_block.max():.3f})")
        #print(f"Area ranges: area=({area_block.min():.3f}, {area_block.max():.3f})")
        # Process each time slice in the block
        results = []
        for i in range(anomaly_block.shape[0]):  # Loop over time dimension
            #print(f"Processing time slice {i}")
            result = process_single_day(anomaly_block[i], area_block, distance_block)
            #print(f"Result shape: {result.shape}, unique values: {np.unique(result)}")
            results.append(result)
        return np.stack(results)
    
    # Apply to your data
    filtered_labels = xr.apply_ufunc(
        wrapper_for_map_blocks,
        ds[anomaly_var], 
        ds[area_var],
        ds['pixel_distance'],
        input_core_dims=[['latitude', 'longitude'], ['latitude', 'longitude'], ['latitude', 'longitude']],
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