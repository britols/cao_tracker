import xarray as xr
import os
from dask.distributed import Client
from utils import climatology

from utils.config import(
    data_path,
    input_for_anomalies,
    output_from_anomalies,
    to_anomaly_var
)

def climatology_test():

    #Set dask client
    memory_limit = '8GB'  # Limit memory usage to 8GB per worker
    client = Client(n_workers=4,memory_limit=memory_limit)  # Starts a local cluster with memory limits
    print(client)

    #Input
    ds = xr.open_dataset("{}{}".format(data_path,input_for_anomalies),chunks={"latitude": "auto", "longitude": 25,"time": -1}) 

    #Calculation
    da = ds[to_anomaly_var]
    ds_anomalies = climatology.scale_da(da)

    #Output
    ds_anomalies['anomaly_scaled'].to_netcdf(os.path.join('{}{}'.format(data_path,output_from_anomalies)),compute=True,mode="w")

    #Close dask client
    client.close()


if __name__ == '__main__':  
    climatology_test()