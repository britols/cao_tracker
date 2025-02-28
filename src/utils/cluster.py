import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage

def array_to_xarray(array,xarray):
    """
    scipy.ndimage converts xarray to array, this function reverts it back to xarray
    """
    new_xarray = xr.DataArray(
        array,
        dims=xarray.dims,  
        coords=xarray.coords,  
        attrs=xarray.attrs,
    )
    return new_xarray

def area_weights(data_array,latitude_dim_name='latitude',R = 6371):
    """
    Area of one grid cell (assuming spatial resolution is 0.25 x 0.25)
    """
    #R = 6371 km^2
    return (R**2)*np.deg2rad(0.25)*np.deg2rad(0.25)*np.cos(np.deg2rad(data_array[latitude_dim_name]))

def label_clusters(data_array, structure=None):
    """
    Defines clusters in data_array and labels it. Structure can be given to change how ndimage clusters points together.
    """
    labeled_array, num_features = ndimage.label(data_array.values, structure=structure) #num_features is ignored
    labeled_xarray = array_to_xarray(labeled_array,data_array)
    return labeled_xarray

def label_and_filter(ds,mask_dim='mask',area_dim='areas',label_dim='labeled_clusters',label_filtered_dim='labeled_clusters_filtered',AREA_THRESHOLD=500000):
    """
    Filters labeled clusters which area is smaller than AREA_THRESHOLD
    """
    ds[label_dim] = label_clusters(ds[mask_dim])
    ds["has_clusters"] = False
    clusters_label = np.unique(ds[label_dim].values)
    areas = ndimage.sum_labels(ds[area_dim],ds[label_dim],clusters_label)
    cluster_pd = pd.DataFrame({"label": clusters_label,"area": areas})
    cluster_pd = cluster_pd[cluster_pd['area']>AREA_THRESHOLD]
    cluster_pd = cluster_pd[cluster_pd['label']>0]
    if not cluster_pd.empty:
        cluster_pd = cluster_pd.reset_index()
        cluster_pd = cluster_pd.drop(['index'],axis=1,inplace=False)
        ds[label_filtered_dim] = ds[label_dim]*ds[label_dim].isin(cluster_pd['label'])
        ds["has_clusters"] = True
    else:
        ds[label_filtered_dim]=np.minimum(ds[label_dim], 0)
    return ds

def get_cluster_info(ds,label_dim="labeled_clusters",label_filtered_dim='labeled_clusters_filtered',anomaly_dim="anomaly_scaled",area_dim='areas',time_dim='time'):
    """
    Returns a pandas data frame cointaining the information below about a cluster:
        label: cluster label
        area: cluster area (in km^2)
        time: time step from ds
        mean: mean value of layer anomaly_dim
        stdev: standard deviation of layer anomaly_dim
        median: median of layer anomaly_dim
        min_value: minimum value of layer anomaly_dim
        min_lat,min_lon: indices of the minimum value of layer anomaly_dim coordinate (ds.longitude[min_lon],ds.latitude[min_lat])
        cm_lat,cm_lon: indices of the center of mass coordinate (ds.longitude[cm_lon],ds.latitude[cm_lat])
    """
    #-------------------------
    clusters_label = np.unique(ds[label_filtered_dim].values)
    areas = ndimage.sum_labels(ds[area_dim],ds[label_dim],clusters_label)
    #-------------------------
    cluster_pd = pd.DataFrame({"label": clusters_label,"area": areas})
    cluster_pd = cluster_pd[cluster_pd['label']>0]
    cluster_pd = cluster_pd.reset_index()
    cluster_pd = cluster_pd.drop(['index'],axis=1,inplace=False)
    cluster_pd['time'] = ds[time_dim].values
    #-------------------------
    clusters_label = np.unique(cluster_pd['label'])
    minimum = ndimage.minimum(ds[anomaly_dim],ds[label_dim],clusters_label)
    stdev = ndimage.standard_deviation(ds[anomaly_dim],ds[label_dim],clusters_label)
    median = ndimage.median(ds[anomaly_dim],ds[label_dim],clusters_label)
    mean = ndimage.mean(ds[anomaly_dim],ds[label_dim],clusters_label)
    #-------------------------
    cm = ndimage.center_of_mass(ds[anomaly_dim].values,ds[label_dim].values,clusters_label)
    cluster_pd[['cm_lat','cm_lon']] = cm #center of mass returns (latitude,longitude)
    #-------------------------
    cluster_pd['mean']= mean
    cluster_pd['stdev']= stdev
    cluster_pd['median']= median
    cluster_pd['min_value']= minimum
    #-------------------------
    cmin = ndimage.minimum_position(ds[anomaly_dim].values,ds[label_dim].values,clusters_label)
    cluster_pd[['min_lat','min_lon']] = cmin #center of mass returns (latitude,longitude)
    cluster_pd['cm_lat'] = cluster_pd['cm_lat'].astype(int)
    cluster_pd['cm_lon'] = cluster_pd['cm_lon'].astype(int)

    return cluster_pd

def apply_binary_morph(data_array,s=np.ones((3,3)),method='dilation'):
    """
    Apply binary structure (s) to data_array using one of the methods available ['dilation','erosion','closing','fill_holes']
    data_array and s must have the same number of dimensions
    """
    if len(data_array.shape) != len(s.shape):
        print("No morphology change: data array and structure have different shapes")
        return(data_array)
    if method not in ['dilation','erosion','closing','fill_holes']:
        print("No morphology change: methods needs to be one of ['dilation','erosion','closing','fill_holes']")
        return(data_array)
    if method=='dilation':
        binary_transformation = ndimage.binary_dilation(data_array,structure=s).astype(int)
    elif method =='erosion':
        binary_transformation = ndimage.binary_erosion(data_array,structure=s).astype(int)
    elif method == 'closing':
        binary_transformation = ndimage.binary_closing(data_array,structure=s).astype(int)
    elif method == 'fill_holes':
        binary_transformation = ndimage.binary_fill_holes(data_array,structure=s).astype(int)
    return array_to_xarray(array=binary_transformation,xarray=data_array)

def gaussian_filter(data_set,center_lon,center_lat,b=0.05,lat_dim_name="latitude",lon_dim_name="longitude"):
    """
    Returns gaussian weights centered on coordinate [center_lon,center_lat], parameter b controls tail's size
    Expects center_lon and center_lat in degrees
    """
    X=data_set[lon_dim_name].broadcast_like(data_set)
    Y=data_set[lat_dim_name].broadcast_like(data_set)
    G = 1/(2*np.pi*b**2) * np.exp(-((np.deg2rad(Y)-np.deg2rad(center_lat))**2 + (np.deg2rad(X)-np.deg2rad(center_lon))**2)/(2*b**2))
    return(G)