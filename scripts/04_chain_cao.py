import xarray as xr
import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
from dask.distributed import Client
from utils import cluster
from utils.config import TrackerConfig

def get_cluster_info_for_day(time_idx, ds_ref):
    """Process one day - this will run on dask workers"""
    try:
        day_data = ds_ref.isel(time=time_idx).load()
        return cluster.get_cluster_info(day_data)
    except Exception as e:
        print(f"Error processing day {time_idx}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

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
    chain_info = identify_cao_chains(cluster_info_df)
    print(f"Found {len(chain_info)} chains")
    
    # Step 2: Calculate intensity metrics for each chain
    print("Calculating chain intensity metrics...")
    chain_metrics = calculate_chain_intensity_metrics(chain_info, cluster_info_df)
    
    # Step 3: Add chain_id back to cluster dataframe
    print("Adding chain IDs to cluster data...")
    cluster_with_chains = add_chain_id_to_clusters(cluster_info_df, chain_info)
    
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