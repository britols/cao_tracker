import xarray as xr
import os
from dask.distributed import Client
from utils import cluster
from utils.config import TrackerConfig

if __name__ == "__main__":

    config = TrackerConfig()
    #if os.path.exists("config.yaml"):
    #    config = config.from_yaml("config.yaml")
    #config = TrackerConfig().from_yaml("config_mod.yaml")

    nc_folder = config.paths.nc_folder
    stdev_file = config.files.stdev_file
    anomaly_file = config.files.anomaly_file #"test_data.nc"
    masked_anomaly_file = config.files.masked_anomaly_file
    scaled_anomaly_var = config.dataset.scaled_anomaly_var
    temperature_var = config.algorithm.temperature_var
    scaled_anomaly_var = config.dataset.scaled_anomaly_var
     
    min_stdev_threshold = config.algorithm.min_stdev_threshold

    memory_limit = '5GB'  # Limit memory usage to 8GB per worker
    client = Client(n_workers=12,memory_limit=memory_limit)  # Starts a local cluster with memory limits
    print("Running dask at ",client.dashboard_link)

    stdev = xr.open_dataset(f"{nc_folder}/{stdev_file}")
    da = stdev[temperature_var]
    mask = da >= min_stdev_threshold
    mask = cluster.apply_binary_morph(mask,method='dilation')
    mask = cluster.apply_binary_morph(mask,method='fill_holes')

    ds = xr.open_dataset(f"{nc_folder}/{anomaly_file}")

    ds[scaled_anomaly_var] = ds[scaled_anomaly_var].where(mask,0)

    foutput_anomaly = f"{nc_folder}/{masked_anomaly_file}"
    ds.to_netcdf(foutput_anomaly, compute=True,mode="w")

    client.close()



# land_sea_mask = xr.open_dataarray('data/IMERG_land_sea_mask.nc')

# # Convert 0-360 longitude to -180 to 180
# land_sea_mask = land_sea_mask.assign_coords(
#     lon=(((land_sea_mask.lon + 180) % 360) - 180)
# ).sortby('lon').rename({'lat': 'latitude', 'lon': 'longitude'})

# reference_array = ds.isel(time=0)

# land_sea_mask = land_sea_mask.interp(
#     latitude=reference_array['labeled_clusters_filtered'].latitude,
#     longitude=reference_array['labeled_clusters_filtered'].longitude,
#     method='nearest'  # or 'nearest', 'cubic'
# )

# mask=land_sea_mask<100
# print('masking')
# ds[var_name] = ds[var_name].where(mask,0)
# print('end masking')