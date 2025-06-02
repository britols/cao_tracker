import xarray as xr
import os
from dask.distributed import Client
from utils import climatology
from utils.config import TrackerConfig

if __name__ == "__main__":

    config = TrackerConfig()
    #if os.path.exists("config.yaml"):
    #    config = config.from_yaml("config.yaml")
    #config = TrackerConfig().from_yaml("config_mod.yaml")

    dir_nc_data = config.paths.era5_path
    temperature_var = config.algorithm.temperature_var
    anomaly_var = config.dataset.anomaly_var
    scaled_anomaly_var = config.dataset.scaled_anomaly_var

    foutput_mean = os.path.join('{}{}'.format(config.paths.nc_folder,config.files.mean_file))
    foutput_stdev = os.path.join('{}{}'.format(config.paths.nc_folder,config.files.stdev_file))
    foutput_anomaly = os.path.join('{}{}'.format(config.paths.nc_folder,config.files.anomaly_file))
    
    memory_limit = '5GB'  # Limit memory usage to 8GB per worker
    client = Client(n_workers=12,memory_limit=memory_limit)  # Starts a local cluster with memory limits
    print("Running dask at ",client.dashboard_link)

    ds = xr.open_mfdataset(os.path.join(dir_nc_data,"*.nc"))
    ds = ds.sel(latitude=slice(70, 20))

    ds = climatology.select_months(ds)
    ds_mean = climatology.mean_calc(da = ds[temperature_var])
    ds_stdev = climatology.std_dev_calc(da = ds[temperature_var])
    ds[anomaly_var], ds[scaled_anomaly_var] = climatology.scale_da(da = ds[temperature_var],season_mean = ds_mean,season_stdev = ds_stdev)

    #ds, encoding = climatology.set_chunks(ds, chunksizes = (1, 201, 441))

    ds.to_netcdf(foutput_anomaly, compute=True,mode="w")#,encoding=encoding)
    ds_mean.to_netcdf(foutput_mean, compute=True,mode="w")
    ds_stdev.to_netcdf(foutput_stdev, compute=True,mode="w")
    
    client.close()




