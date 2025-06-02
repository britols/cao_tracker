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

def get_cluster_info_for_day(time_idx, ds_ref):
    """Process one day - this will run on dask workers"""
    try:
        day_data = ds_ref.isel(time=time_idx).load()
        return cluster.get_cluster_info(day_data)
    except Exception as e:
        print(f"Error processing day {time_idx}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

if __name__ == "__main__":
    
    config = TrackerConfig()
    
    var_name = config.dataset.label_filtered_var
    labeled_nc_folder = config.paths.labeled_nc_folder
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
    
    # Extract cluster information using joblib + dask (exactly like your original)
    print("Processing with joblib + dask...")
    with joblib.parallel_config(backend="dask"):
        dataframes = joblib.Parallel(verbose=10)(
            joblib.delayed(get_cluster_info_for_day)(i, ds) 
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

    # Step 1: Identify chains of consecutive CAO days
    print("Identifying CAO chains...")
    chain_info = cluster.identify_cao_chains(cluster_info_df)
    print(f"Found {len(chain_info)} chains")
    
    # Step 2: Calculate intensity metrics for each chain
    print("Calculating chain intensity metrics...")
    chain_metrics = cluster.calculate_chain_intensity_metrics(chain_info, cluster_info_df)
    
    # Step 3: Add chain_id back to cluster dataframe
    print("Adding chain IDs to cluster data...")
    cluster_with_chains = cluster.add_chain_id_to_clusters(cluster_info_df, chain_info)
    
    # Save results
    chain_metrics.to_csv(f"{csv_folder}/{cao_chain_table}", index=False)
    cluster_with_chains.to_csv(f"{csv_folder}/clusters_with_chains.csv", index=False)
    
    print(f"Saved chain metrics to {cao_chain_table}")
    print(f"Saved clusters with chain IDs to clusters_with_chains.csv")
    
    # Print summary statistics
    print("\n=== CHAIN SUMMARY ===")
    print(f"Total chains identified: {len(chain_metrics)}")
    print(f"Chain duration range: {chain_metrics['duration_days'].min()}-{chain_metrics['duration_days'].max()} days")
    print(f"Mean chain duration: {chain_metrics['duration_days'].mean():.1f} days")
    print(f"Longest chain: {chain_metrics['duration_days'].max()} days")
    print(f"Max area in any chain: {chain_metrics['max_area_km2'].max():.0f} km²")
    print(f"Coldest temperature: {chain_metrics['min_temperature'].min():.2f} (scaled anomaly)")
    
    client.close()