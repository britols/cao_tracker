#C:\Users\ls2236\AppData\Local\anaconda3\envs\arco\python.exe C:\Users\ls2236\Projects\BIG\ERA5\arco-era5\download\download_temperature.py
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from distributed import Client,LocalCluster
import time
from scipy import ndimage
import matplotlib.colors
import os
from matplotlib.colors import TwoSlopeNorm
import datetime


def load_data(dir_data):

    ncdata_full = xr.open_mfdataset(os.path.join(dir_data,"*.nc"))
    ncdata_var = ncdata_full["daily_t2_min"]
    return(ncdata_var)

def calc_stats(ncdata):

    tmean_season = ncdata.groupby("time.season").mean(dim='time').persist()
    tmean_season_std = ncdata.groupby("time.season").std(dim='time').persist()
    anomaly = (t2min_raw - tmean_season.sel(season='DJF')).persist()
    standard_anomaly =anomaly/tmean_season_std.sel(season='DJF').persist()
    return(standard_anomaly,tmean_season_std,anomaly)


def process_time_slice_and_plot(time_slice, anom_slice,stdev_slice,time_coord, lat_vals, lon_vals, size_threshold=1750):

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["purple","darkblue","blue","lightblue","white","lightcoral","red","darkred","pink"])
    z = [-35,-30,-25,-20,-15,-10,-5,0,1,10,20,50]
    norm = TwoSlopeNorm(vmin=np.min(z), vcenter=0, vmax=np.max(z))

    labeled_slice, num_features = ndimage.label(time_slice.values)
    region_sizes = np.bincount(labeled_slice.ravel())
    region_sizes[0] = 0  # Ignore background
    largest_region_label = np.argwhere(region_sizes >= size_threshold)
    if np.size(largest_region_label) > 0:
        largest_region_size = region_sizes[largest_region_label]
        if len(region_sizes) > 1:
            lab_list = []
            for lab in largest_region_label:
                lab_list.append([anom_slice.where(labeled_slice == lab).quantile(0.25).values])
            loc = lab_list <= lab_list[np.argmax(region_sizes[largest_region_label])]+(region_sizes[largest_region_label] - np.max(region_sizes[largest_region_label]))/5000
            region_info = pd.DataFrame(lab_list,columns=["anomaly"])
            region_info['labs'] = largest_region_label
            region_info['Nsize'] = region_sizes[largest_region_label]
            region_info =region_info.loc[loc]
            region_info=region_info.reset_index()
            print(region_info)
            largest_region_label = region_info['labs'][region_info["anomaly"].argmin()]
            #largest_region_label = largest_region_label[np.argmin(lab_list)]
            largest_region_size = region_sizes[largest_region_label]
        temp_array = xr.DataArray(labeled_slice,coords={"latitude": time_slice["latitude"].values,"longitude": time_slice["longitude"].values},dims=['latitude','longitude'])
        temp2=stdev_slice.where(labeled_slice == largest_region_label,0)
        anom_min=anom_slice.where(labeled_slice == largest_region_label,0).min().values
        anom_mean=anom_slice.where(labeled_slice == largest_region_label,0).mean().values
        anom_min_all=anom_slice.min().values
        #com = ndimage.center_of_mass(np.abs(temp2.values))
        com = ndimage.center_of_mass(temp2.values)
        com_lat = lat_vals[int(com[0])]
        com_lon = lon_vals[int(com[1])]
        # Plot the result
        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.contourf(lon_vals, lat_vals, anom_slice,cmap=cmap,norm=norm,levels=z,extend="both")
        plt.contour(lon_vals, lat_vals, temp_array,colors="red")
        plt.contour(lon_vals, lat_vals, temp_array.where(labeled_slice != largest_region_label),colors="black")
        plt.scatter(com_lon, com_lat, color='yellow', marker='x', label='Center of Mass')
        ax.coastlines( edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
        plt.title(f'Time: {str(time_coord)} | Largest Region Size: {largest_region_size}')
        #filename=pd.to_datetime(str(time[t_idx])).strftime("%Y%m%d") 
        filename=pd.to_datetime(str(time[t_idx])).strftime("%Y%m%d") 
        plt.savefig(os.path.join(save_path, f'labeled_region_{str(filename)}.png'))
        plt.close()
        print(time_coord,com_lat,com_lon)
        return com_lat, com_lon, anom_min, anom_min_all,anom_mean,largest_region_size
    # If no valid region, return NaN
    return np.nan, np.nan, np.nan, np.nan, np.nan,np.nan




if __name__ == "__main__":

    cluster = LocalCluster()         
    client = cluster.get_client()
    print(cluster.get_client)

    dir_data = r"C:\Users\ls2236\Projects\BIG\ERA5\arco-era5\data"
    
    t2min_raw = load_data(os.path.join(dir_data,"daily"))
    
    standard_anomaly,tmean_season_std,anomaly = calc_stats(t2min_raw)
    cao_bool = standard_anomaly.where(tmean_season_std.sel(season='DJF') > 3,0)
    cao_bool = cao_bool < -1.75
    save_path = r"C:\Users\ls2236\Projects\BIG\ERA5\arco-era5\cao_tracker\img"

    for year in range(1941,2023): 
    #for year in range(1982,1983): 
        print(year)
        my_list = []
        date_i = f'{year-1}-12-01'
        date_f = f'{year}-03-01'
        data_array = cao_bool.sel(time=slice(date_i,date_f))
        data_array_anom = anomaly.sel(time=slice(date_i,date_f))
        data_array_stdev = standard_anomaly.sel(time=slice(date_i,date_f))
        time = data_array["time"].values
        latitude = data_array["latitude"].values
        longitude = data_array["longitude"].values
        data = np.random.randint(0, 2, size=(len(time), len(latitude), len(longitude)))
    # Loop through time and process each time slice
        for t_idx in range(len(time)):
            time_slice = data_array.isel(time=t_idx)
            anom_slice = data_array_anom.isel(time=t_idx)
            stdev_slice = data_array_stdev.isel(time=t_idx)
            com_lat, com_lon, anom_min, anom_min_all,anom_mean,largest_region_size = process_time_slice_and_plot(time_slice,anom_slice,stdev_slice, time[t_idx], latitude, longitude)
            my_list.append([time[t_idx],com_lat, com_lon, anom_min, anom_min_all,anom_mean,largest_region_size])
            #print(com_lat, com_lon, anom_min, anom_min_all)
        pd.DataFrame(my_list,columns=["time","lat","lon","min_rel","min_all","anom_mean","size"]).to_csv(os.path.join(dir_data,"tables/{}.csv".format(year)), sep=';',index=False)
    client.close()