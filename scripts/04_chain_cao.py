import xarray as xr
import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
from dask.distributed import Client
from utils import cluster
from utils.config import TrackerConfig
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in divide", 
                       category=RuntimeWarning, module="scipy.ndimage")

def get_cluster_info_for_day(time_idx, ds_ref, ocean_mask=None, apply_ocean_mask=False):
    """Process one day - this will run on dask workers"""
    try:
        day_data = ds_ref.isel(time=time_idx).load()
        return cluster.get_cluster_info(day_data,apply_ocean_mask=apply_ocean_mask, 
                                      ocean_mask=ocean_mask)
    except Exception as e:
        print(f"Error processing day {time_idx}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

if __name__ == "__main__":
    
    config = TrackerConfig()
    
    var_name = config.dataset.label_filtered_var
    nc_folder = config.paths.data_path
    labeled_nc_folder = config.paths.labeled_nc_folder
    ocean_mask_file = config.files.ocean_mask
    csv_folder = config.paths.csv_folder
    cao_cluster_table = config.files.cao_cluster_table
    cao_chain_table = config.files.cao_chain_table

    memory_limit = '5GB'
    client = Client(n_workers=12, memory_limit=memory_limit)
    print("Running dask at", client.dashboard_link)

    # Check if cluster table already exists to avoid reprocessing
    cluster_table_path = f"{csv_folder}/{cao_cluster_table}"
    
    print("Processing from NetCDF files...")
    
    # Load dataset - same pattern as your original script
    print("Loading dataset...")
    ds = xr.open_mfdataset(f"{labeled_nc_folder}/*.nc")
    #ds = ds.sel(time = slice(str('1980'),str('1983')))
    ocean_mask = xr.open_dataarray(f"{nc_folder}{ocean_mask_file}")

    # Extract cluster information using joblib + dask (exactly like your original)
    print("Processing with joblib + dask...")
    with joblib.parallel_config(backend="dask"):
        dataframes = joblib.Parallel(verbose=10)(
            joblib.delayed(get_cluster_info_for_day)(i, ds,ocean_mask=ocean_mask, apply_ocean_mask=True) 
            for i in tqdm(range(len(ds.time)))
        )
    
    # Combine results (filter out empty DataFrames)
    cluster_info_df = pd.concat([df for df in dataframes if not df.empty], ignore_index=True)
    
    if not cluster_info_df.empty:
        cluster_info_df['time'] = pd.to_datetime(cluster_info_df['time'])
        
        # Save cluster table
        cluster_info_df.to_csv(cluster_table_path, index=False)
        print(f"Saved cluster information to {cao_cluster_table}")
    else:
        print("No clusters found in dataset!")
        client.close()
        exit()

    # === FILTER BY LAND FRACTION BEFORE CHAINING ===
    print("Filtering clusters by land fraction (≥5% land coverage)...")
    if 'land_fraction' in cluster_info_df.columns:
        cluster_info_land = cluster_info_df[cluster_info_df['land_fraction'] >= 0.05].copy()
        cluster_info_land = cluster_info_land.reset_index(drop=True)
        
        print(f"Kept {len(cluster_info_land)}/{len(cluster_info_df)} clusters after land filtering")
        
        # Save land-filtered cluster table
        cluster_info_land.to_csv(f"{csv_folder}/cao_cluster_table_land.csv", index=False)
    else:
        print("Warning: No land_fraction column found. Using all clusters.")
        cluster_info_land = cluster_info_df.copy()

    # Step 1: Identify chains of consecutive CAO days
    print("Identifying CAO chains...")
    chain_info = cluster.identify_cao_chains(cluster_info_land)
    print(f"Found {len(chain_info)} chains")
    
    # Step 2: Calculate intensity metrics for each chain
    print("Calculating chain intensity metrics...")
    chain_metrics = cluster.calculate_chain_intensity_metrics(chain_info, cluster_info_land)
    
    # Step 3: Add chain_id back to cluster dataframe
    print("Adding chain IDs to cluster data...")
    cluster_with_chains = cluster.add_chain_id_to_clusters(cluster_info_land, chain_info)

    # Save results
    chain_metrics.to_csv(f"{csv_folder}/{cao_chain_table}", index=False)
    cluster_with_chains.to_csv(f"{csv_folder}/clusters_with_chains.csv", index=False)
    
    # Print summary statistics
    print("\n=== CHAIN SUMMARY (LAND-FILTERED) ===")
    print(f"Total chains identified: {len(chain_metrics)}")
    if len(chain_metrics) > 0:
        print(f"Chain duration range: {chain_metrics['duration_days'].min()}-{chain_metrics['duration_days'].max()} days")
        print(f"Mean chain duration: {chain_metrics['duration_days'].mean():.1f} days")
        print(f"Longest chain: {chain_metrics['duration_days'].max()} days")
        print(f"Max area in any chain: {chain_metrics['max_area_km2'].max():.0f} km²")
        print(f"Coldest temperature: {chain_metrics['min_temperature'].min():.2f} (scaled anomaly)")
    
    
    client.close()