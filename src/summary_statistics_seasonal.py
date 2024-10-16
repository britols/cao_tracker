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
    #Calculate seasonal average
    tmean_season = t2min_raw.groupby("time.season").mean(dim='time').persist()
    #Calculate seasonal standard deviation
    tmean_season_std = t2min_raw.groupby("time.season").std(dim='time').persist()

    #tmean_season.to_netcdf(os.path.join(save_path, 'Seasonal_daily_mean.nc'),compute=True,mode="w")
    #tmean_season_std.to_netcdf(os.path.join(save_path, 'Seasonal_daily_stdev.nc'),compute=True,mode="w")

    #Calculate daily anomaly
    anomaly = (t2min_raw - tmean_season.sel(season='DJF')).persist()#.to_netcdf(os.path.join(save_path, 'Seasonal_daily_anomaly.nc'),compute=True)


    #Daily Standarized anomalies
    ((anomaly)/tmean_season_std.sel(season='DJF')).to_netcdf(os.path.join(save_path, 'Seasonal_daily_standard_anomaly.nc'),compute=True,mode="w")
    


    client.close()

    #tmean_daily_roll = ts.rolling(time=31, center=True).mean(dim='time').compute().groupby("time.dayofyear").mean(dim='time').to_netcdf(os.path.join(save_path, 'Rolling_daily_mean.nc'),compute=True,mode="w")

    # Calculate daily standard deviation
    #window=15
    #ds_clima = []

    #for doy in range(1, 365):
    #    doys = [365 + doy if doy <= 0 else doy for doy in np.arange(doy - window, doy + window + 1)]
    #    tmp = ts.where(ts.time.dt.dayofyear.isin(doys)).std(dim="time")
    #    ds_clima.append(tmp.assign_coords(doy=doy))

    #ds_out = xr.concat(ds_clima, dim='doy').load()