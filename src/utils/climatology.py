import xarray as xr

def scale_da(da,time_dim = 'time',time_group = 'season',time_group_period = 'DJF'):
    '''
    Calculates mean and standard deviation of data_array and returns anomaly and scaled anomaly 
    default parameters: group 'time' by 'season' and select 'DJF'
    '''
    season_mean = da.groupby("{}.{}".format(time_dim,time_group)).mean(dim=time_dim)
    season_stdev = da.groupby("{}.{}".format(time_dim,time_group)).std(dim=time_dim)
    season_mean = season_mean.sel(season=time_group_period).drop_vars(time_group)
    season_stdev = season_stdev.sel(season=time_group_period).drop_vars(time_group)
    anomaly = (da - season_mean)
    anomaly_scaled = anomaly/season_stdev
    ds_temp = xr.Dataset({'anomaly': anomaly,'anomaly_scaled': anomaly_scaled})
    return(ds_temp)
