from distributed import Client,LocalCluster
import xarray as xr
import os
import dask
#PARAMETERS_DICT = dict(zip(aliases, true_file_names))
#PARAMETERS.keys()
#PARAMETERS.items()

PARAMETERS = {
    "dir_nc_data": r"C:\Users\ls2236\Projects\BIG\ERA5\arco-era5\data\daily",
    "dir_to_save": r"C:\Users\ls2236\Projects\CAO_TRACKER\data\statistics_and_transformed",
    "nc_variable": "daily_t2_min"
}

if __name__ == "__main__":

    cluster = LocalCluster()         
    client = cluster.get_client()
    print(cluster.get_client)

    save_path = PARAMETERS["dir_to_save"]

    #load ERA5 netcdf data
    era5_data = xr.open_mfdataset(os.path.join(PARAMETERS["dir_nc_data"],"*.nc"))
    #Select variable
    t2min_raw = era5_data[PARAMETERS["nc_variable"]]
    #t2min_raw =  t2min_raw.groupby("time.dayofyear")

    #anomaly
    #-------
    #rolling_mean = xr.open_dataset(r"C:\Users\ls2236\Projects\CAO_TRACKER\data\statistics_and_transformed\Rolling_daily_mean.nc")

    #gb = t2min_raw.groupby('time.dayofyear')
    #clim = rolling_mean["daily_t2_min"]
    #anom = gb - clim
    #anom.to_netcdf(os.path.join(save_path, 'Rolling_daily_anomaly.nc'),compute=True)
    #standarized anomaly
    #-------
    rolling_stdev = xr.open_dataset(r"C:\Users\ls2236\Projects\CAO_TRACKER\data\statistics_and_transformed\Rolling_daily_stdev.nc")
    anomaly = xr.open_dataset(r"C:\Users\ls2236\Projects\CAO_TRACKER\data\statistics_and_transformed\Rolling_daily_anomaly.nc")
    gb = anomaly.groupby('time.dayofyear')
    anom_stdev = gb/rolling_stdev["daily_t2_min"]
    anom_stdev.to_netcdf(os.path.join(save_path, 'Rolling_daily_standard_anomaly.nc'),compute=True)

    client.close()
