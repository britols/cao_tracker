import xarray as xr

def select_months(ds,months=[11, 12, 1, 2, 3]):
    '''
    Selects the months from the cold season (don't need to be DJF), by default selects november to march
    '''
    ds = ds.sel(time=ds.time.dt.month.isin(months))
    return(ds)

def mean_calc(da,time_dim = 'time',time_group = 'season',time_group_period = 'DJF'):
    '''
    Calculates the seasonal mean of a data array, by default the DJF mean
    '''
    season_mean = da.groupby("{}.{}".format(time_dim,time_group)).mean(dim=time_dim)
    season_mean = season_mean.sel(season=time_group_period).drop_vars(time_group)
    return(season_mean)

def std_dev_calc(da,time_dim = 'time',time_group = 'season',time_group_period = 'DJF'):
    '''
    Calculates the seasonal standard deviation of a data array, by default the DJF standard deviation
    '''
    season_stdev = da.groupby("{}.{}".format(time_dim,time_group)).std(dim=time_dim)
    season_stdev = season_stdev.sel(season=time_group_period).drop_vars(time_group)
    return(season_stdev)

def scale_da(da,season_mean,season_stdev):
    '''
    Calculates the temperature anomaly and scaled anomaly fields
    '''
    anomaly = (da - season_mean)
    anomaly_scaled = anomaly/season_stdev
    return(anomaly,anomaly_scaled)

def set_chunks (ds, chunksizes = (1, 281, 441)):
    '''
    Chunk a given dataset according to chunksizes. Returns the chunked dataset and encondig, 
    which can be used in .to_netcdf(encoding=encoding)
    '''
    ds = ds.chunk({'time': chunksizes[0], 'latitude': chunksizes[1], 'longitude': chunksizes[2]})
    encoding = {var: {'chunksizes': chunksizes} for var in ds.data_vars}
    return(ds,encoding)
    #ds.to_netcdf(foutput, compute= to_compute,mode="w",encoding=encoding)