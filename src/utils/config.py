"""
Set paths
"""
data_path = 'data/'
test_path = 'test/'
image_folder = 'img/'
csv_folder = 'csv/'
to_anomaly_var = "daily_t2_min"
"""
Set files
"""
input_for_anomalies = "era5_t2min_1970_2000.nc"
output_from_anomalies = "era5_t2min_scaled_anomalies.nc"
input_for_clusters = "era5_t2min_scaled_anomalies.nc"
output_from_clusters = "era5_t2min_clusters.nc"
output_from_clusters_table = "era5_t2min_clusters.csv"
"""
Set Thresholds
"""
STDEV_THRESHOLD = -1.5
AREA_THRESHOLD=500000#in km^2
"""
Dataset layers names
"""
time_dim = 'time'
time_group = 'season'
time_group_period = 'DJF'
latitude_dim_name='latitude'
mask_dim='mask'
area_dim='areas'
label_dim="labeled_clusters"
label_filtered_dim='labeled_clusters_filtered'
anomaly_dim="anomaly_scaled"
