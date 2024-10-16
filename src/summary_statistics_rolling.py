from distributed import Client,LocalCluster
import xarray as xr
import numpy as np
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

    daily_t2_min = np.zeros([366,281,441])
    lon = t2min_raw.longitude.values
    lat = t2min_raw.latitude.values
    time =range(1,367,1)
    
    da = xr.Dataset(
    data_vars=dict(
        daily_t2_min=(["dayofyear", "latitude","longitude"],daily_t2_min)
    ),
    #dims=[ "dayofyear", "latitude","longitude",],
    coords=dict(
        dayofyear=time,
        latitude=(["latitude"], lat),
        longitude=(["longitude"], lon)
    ),
    attrs=dict(
        long_name="2 metre temperature",
        short_name="t2m",
        unit="K"
    ))

    #tmean_daily_roll = t2min_raw.rolling(time=31, center=True).mean(dim='time').persist()
    #tmean_daily_roll = tmean_daily_roll.groupby("time.dayofyear").mean(dim='time')
    #tmean_daily_roll.to_netcdf(os.path.join(save_path, 'Rolling_daily_mean.nc'),compute=True,mode="w")

    # Calculate daily standard deviation
    window=15

    for doy in range(1, 367,1):
        #doys = [365 + doy if doy <= 0 else doy for doy in np.arange(doy - window, doy + window + 1)]
        doys=np.arange(doy - window, doy + window + 1)
        doys=list(map(lambda x: 366+x if (x<=0) else x, doys))
        doys=list(map(lambda x: x-366 if (x>366) else x, doys))
        print(doy)
        tmp = t2min_raw.where(t2min_raw.time.dt.dayofyear.isin(doys)).std(dim="time")
        da.loc[dict(dayofyear=doy)] = tmp
    da.to_netcdf(os.path.join(save_path, 'Rolling_daily_stdev.nc'),compute=True,mode="w")
    client.close()