import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage

def array_to_xarray(array,xarray):
    """
    scipy.ndimage converts xarray to array, this function reverts it back to xarray
    """
    new_xarray = xr.DataArray(
        array,
        dims=xarray.dims,  
        coords=xarray.coords,  
        attrs=xarray.attrs,
    )
    return new_xarray

def area_weights(data_array,latitude_dim_name='latitude',R = 6371,lon_res=0.25,lat_res=0.25):
    """
    Area of one grid cell (assuming spatial resolution is 0.25 x 0.25)
    """
    #R = 6371 km^2
    return (R**2)*np.deg2rad(lon_res)*np.deg2rad(lat_res)*np.cos(np.deg2rad(data_array[latitude_dim_name]))

def calculate_grid_distances(data_array,latitude_dim_name='latitude', R=6371,lon_res=0.25,lat_res=0.25):
  
    lon_distance = R * np.deg2rad(lon_res) * np.cos(np.deg2rad(data_array[latitude_dim_name]))
    lat_distance =  R * np.deg2rad(lat_res) * xr.ones_like(data_array[latitude_dim_name])

    return lon_distance, lat_distance  

def label_filter_and_merge(anomaly_data, area_data,  distance_data=None,threshold=-1.5, area_threshold=500000, distance_threshold = 1000):
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
        filtered_result = merge_nearby_labels_2d(filtered_result, distance_array=distance_data,distance_threshold_km=distance_threshold)

    return filtered_result

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

def get_cluster_info(ds,label_dim="labeled_clusters",label_filtered_dim='labeled_clusters_filtered',
                     anomaly_dim="scaled_anomaly",area_dim='areas',time_dim='time',
                     apply_ocean_mask=False, ocean_mask=None):
    """
    Returns a pandas data frame cointaining the information below about a cluster:
        label: cluster label
        area: cluster area (in km^2)
        time: time step from ds
        mean: mean value of layer anomaly_dim
        stdev: standard deviation of layer anomaly_dim
        median: median of layer anomaly_dim
        min_value: minimum value of layer anomaly_dim
        min_lat,min_lon: indices of the minimum value of layer anomaly_dim coordinate (ds.longitude[min_lon],ds.latitude[min_lat])
        cm_lat,cm_lon: indices of the center of mass coordinate (ds.longitude[cm_lon],ds.latitude[cm_lat])
    """
    # Define all columns upfront (your schema)
    columns = [
        'time', 'label', 'area', 'cm_lat', 
        'cm_lon', 'mean', 'stdev', 'median', 
        'min_value','min_lat','min_lon'
    ]
    if apply_ocean_mask:
        columns.append('land_fraction')
    # Create empty DataFrame with correct structure
    if not ds['has_clusters'].values:
        cluster_pd = pd.DataFrame(columns=columns)
        return cluster_pd  # Return empty but structured DataFrame
    if len(np.unique(ds[label_filtered_dim])) <= 1:
        cluster_pd = pd.DataFrame(columns=columns)
        return cluster_pd  # Return empty but structured DataFrame
    #-------------------------
    #cluster_pd['label'] = np.unique(ds[label_filtered_dim].values)
    #cluster_pd['area'] = ndimage.sum_labels(ds[area_dim],ds[label_filtered_dim],clusters_label)
    clusters_label = np.unique(ds[label_filtered_dim].values)
    areas = ndimage.sum_labels(ds[area_dim],ds[label_filtered_dim],clusters_label)
    #-------------------------
    cluster_pd = pd.DataFrame({"label": clusters_label,"area": areas})
    cluster_pd = cluster_pd[cluster_pd['label']>0]
    cluster_pd = cluster_pd.reset_index()
    cluster_pd = cluster_pd.drop(['index'],axis=1,inplace=False)
    cluster_pd['time'] = ds[time_dim].values
    #-------------------------
    clusters_label = np.unique(cluster_pd['label'])
    minimum = ndimage.minimum(ds[anomaly_dim],ds[label_filtered_dim],clusters_label)
    stdev = ndimage.standard_deviation(ds[anomaly_dim],ds[label_filtered_dim],clusters_label)
    median = ndimage.median(ds[anomaly_dim],ds[label_filtered_dim],clusters_label)
    mean = ndimage.mean(ds[anomaly_dim],ds[label_filtered_dim],clusters_label)
    #-------------------------
    cm = ndimage.center_of_mass(ds[anomaly_dim].values,ds[label_filtered_dim].values,clusters_label)
    cluster_pd[['cm_lat','cm_lon']] = cm #center of mass returns (latitude,longitude)
    #-------------------------
    cluster_pd['mean']= mean
    cluster_pd['stdev']= stdev
    cluster_pd['median']= median
    cluster_pd['min_value']= minimum
    #-------------------------
    cmin = ndimage.minimum_position(ds[anomaly_dim].values,ds[label_filtered_dim].values,clusters_label)
    cluster_pd[['min_lat','min_lon']] = cmin #center of mass returns (latitude,longitude)
    cluster_pd['cm_lat'] = cluster_pd['cm_lat'].astype(int)
    cluster_pd['cm_lon'] = cluster_pd['cm_lon'].astype(int)

    # === NEW: Ocean mask calculations ===
    if apply_ocean_mask and ocean_mask is not None:
        # Calculate land areas for each cluster
        land_areas = ndimage.sum_labels(
            ocean_mask.values, 
            ds[label_filtered_dim].values, 
            clusters_label
        )
        
        # Calculate total pixel counts for each cluster (for land fraction)
        pixel_counts = ndimage.sum_labels(
            np.ones_like(ds[label_filtered_dim].values), 
            ds[label_filtered_dim].values, 
            clusters_label
        )
        
        # Calculate land fraction for each cluster
        land_fractions = land_areas / pixel_counts
        
        # Add land fraction to the dataframe
        cluster_pd['land_fraction'] = land_fractions

    return cluster_pd

def identify_cao_chains(cluster_df):
    """
    Identify chains of consecutive days with CAO activity
    Returns DataFrame with chain information
    """
    # Get unique dates and sort them
    dates = sorted(cluster_df['time'].dt.date.unique())
    
    chains = []
    current_chain = []
    chain_id = 1
    
    for i, date in enumerate(dates):
        if not current_chain:
            # Start new chain
            current_chain = [date]
        else:
            # Check if current date is consecutive to last date in chain
            prev_date = current_chain[-1]
            if (date - prev_date).days == 1:
                # Continue current chain
                current_chain.append(date)
            else:
                # End current chain and start new one
                if len(current_chain) > 0:
                    chains.append({
                        'chain_id': chain_id,
                        'start_date': current_chain[0],
                        'end_date': current_chain[-1],
                        'duration_days': len(current_chain),
                        'dates': current_chain.copy()
                    })
                    chain_id += 1
                current_chain = [date]
    
    # Don't forget the last chain
    if len(current_chain) > 0:
        chains.append({
            'chain_id': chain_id,
            'start_date': current_chain[0],
            'end_date': current_chain[-1],
            'duration_days': len(current_chain),
            'dates': current_chain.copy()
        })
    
    return pd.DataFrame(chains)

def calculate_chain_intensity_metrics(chain_info, cluster_df):
    """
    Calculate intensity metrics for each chain
    """
    chain_metrics = []
    
    for _, chain in chain_info.iterrows():
        # Get all clusters for this chain's date range
        chain_dates = pd.to_datetime(chain['dates']).date
        chain_clusters = cluster_df[cluster_df['time'].dt.date.isin(chain_dates)]
        
        if chain_clusters.empty:
            continue
            
        # Calculate intensity metrics
        metrics = {
            'chain_id': chain['chain_id'],
            'start_date': chain['start_date'],
            'end_date': chain['end_date'],
            'duration_days': chain['duration_days'],
            
            # Area metrics
            'max_area_km2': chain_clusters['area'].max(),
            'mean_area_km2': chain_clusters['area'].mean(),
            'total_area_km2': chain_clusters['area'].sum(),
            
            # Temperature metrics (scaled anomaly - more negative = colder)
            'min_temperature': chain_clusters['min_value'].min(),  # Coldest temperature
            'mean_min_temperature': chain_clusters['min_value'].mean(),
            'mean_temperature': chain_clusters['mean'].mean(),
            
            # Spatial extent
            'max_lat': chain_clusters['cm_lat'].max(),
            'min_lat': chain_clusters['cm_lat'].min(),
            'max_lon': chain_clusters['cm_lon'].max(),
            'min_lon': chain_clusters['cm_lon'].min(),
            
            # Cluster count metrics
            'total_clusters': len(chain_clusters),
            'max_clusters_per_day': chain_clusters.groupby(chain_clusters['time'].dt.date).size().max(),
            'mean_clusters_per_day': chain_clusters.groupby(chain_clusters['time'].dt.date).size().mean(),
            
            # Additional metrics
            'max_cluster_area_day': chain_clusters.loc[chain_clusters['area'].idxmax(), 'time'].date(),
            'coldest_temp_day': chain_clusters.loc[chain_clusters['min_value'].idxmin(), 'time'].date(),
        }
        
        chain_metrics.append(metrics)
    
    return pd.DataFrame(chain_metrics)

def add_chain_id_to_clusters(cluster_df, chain_info):
    """
    Add chain_id to the original cluster dataframe
    """
    cluster_df['chain_id'] = 0  # Default for clusters not in chains
    
    for _, chain in chain_info.iterrows():
        chain_dates = pd.to_datetime(chain['dates']).date
        mask = cluster_df['time'].dt.date.isin(chain_dates)
        cluster_df.loc[mask, 'chain_id'] = chain['chain_id']
    
    return cluster_df


def apply_binary_morph(data_array,s=np.ones((3,3)),method='dilation'):
    """
    Apply binary structure (s) to data_array using one of the methods available ['dilation','erosion','closing','fill_holes']
    data_array and s must have the same number of dimensions
    """
    if len(data_array.shape) != len(s.shape):
        print("No morphology change: data array and structure have different shapes")
        return(data_array)
    if method not in ['dilation','erosion','closing','fill_holes']:
        print("No morphology change: methods needs to be one of ['dilation','erosion','closing','fill_holes']")
        return(data_array)
    if method=='dilation':
        binary_transformation = ndimage.binary_dilation(data_array,structure=s).astype(int)
    elif method =='erosion':
        binary_transformation = ndimage.binary_erosion(data_array,structure=s).astype(int)
    elif method == 'closing':
        binary_transformation = ndimage.binary_closing(data_array,structure=s).astype(int)
    elif method == 'fill_holes':
        binary_transformation = ndimage.binary_fill_holes(data_array,structure=s).astype(int)
    return array_to_xarray(array=binary_transformation,xarray=data_array)

def gaussian_filter(data_set,center_lon,center_lat,b=0.05,lat_dim_name="latitude",lon_dim_name="longitude"):
    """
    Returns gaussian weights centered on coordinate [center_lon,center_lat], parameter b controls tail's size
    Expects center_lon and center_lat in degrees
    """
    X=data_set[lon_dim_name].broadcast_like(data_set)
    Y=data_set[lat_dim_name].broadcast_like(data_set)
    G = 1/(2*np.pi*b**2) * np.exp(-((np.deg2rad(Y)-np.deg2rad(center_lat))**2 + (np.deg2rad(X)-np.deg2rad(center_lon))**2)/(2*b**2))
    return(G)


def process_in_chunks(ds, label_cao_output_dir,chunk_size_years=10):
    years = pd.to_datetime(ds.time.values).year
    unique_years = np.unique(years)
    
    for i in range(0, len(unique_years), chunk_size_years):
        year_chunk = unique_years[i:i + chunk_size_years]
        start_year, end_year = year_chunk[0], year_chunk[-1]
        
        print(f"Processing years {start_year}-{end_year}...")
        ds_chunk = ds.sel(time=slice(str(start_year), str(end_year)))
        
        # Your processing logic here
        # ... apply clustering, filtering, etc.
        
        # Save results
        ds_chunk.to_netcdf(f"{label_cao_output_dir}/clusters_{start_year}_{end_year}.nc", mode="w")
        
        # Optional: explicitly free memory
        del ds_chunk

def mask_ocean(da_ref,file_in = 'data/IMERG_land_sea_mask.nc',file_out = 'data/ocean_mask.nc'):

    land_sea_mask = xr.open_dataarray(file_in)
    # # Convert 0-360 longitude to -180 to 180
    land_sea_mask = land_sea_mask.assign_coords(
    lon=(((land_sea_mask.lon + 180) % 360) - 180)
    ).sortby('lon').rename({'lat': 'latitude', 'lon': 'longitude'})

    land_sea_mask = land_sea_mask.interp(
    latitude=da_ref.latitude,
    longitude=da_ref.longitude,
    method='nearest'  # or 'nearest', 'cubic'
    )

    mask=land_sea_mask<100

    # Define Greenland boundaries (approximate)
    greenland_lat_min = 59  # degrees N
    greenland_lat_max = 84  # degrees N  
    greenland_lon_min = -75 # degrees W
    greenland_lon_max = -10 # degrees W

    # Create Greenland mask
    greenland_mask = (
        (mask.latitude >= greenland_lat_min) & 
        (mask.latitude <= greenland_lat_max) &
        (mask.longitude >= greenland_lon_min) & 
        (mask.longitude <= greenland_lon_max)
    )

    # Set Greenland to ocean (1) in the mask
    mask = mask.where(~greenland_mask, 0)

    mask.to_netcdf(file_out)
    #print('masking')
    #ds[var_name] = ds[var_name].where(mask,0)
    #print('end masking')