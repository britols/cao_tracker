from distributed import Client,LocalCluster
import pandas as pd
import numpy as np
import datetime
import os
#PARAMETERS_DICT = dict(zip(aliases, true_file_names))
#PARAMETERS.keys()
#PARAMETERS.items()

PARAMETERS = {
    #"dirname": r"C:\Users\ls2236\Projects\BIG\ERA5\arco-era5\data\tables",
    "dirname": r"C:\Users\ls2236\Projects\cao_tracker\results\tracker_daily_stats\tables",
    "distance": 3500,
    "lat_diff_max": 3,
    "output_file": r"C:\Users\ls2236\Projects\cao_tracker\results\tracker_daily_stats\cao_tracker_rolling_1_5_stdev.csv"
    #"output_file": r"C:\Users\ls2236\Projects\BIG\ERA5\arco-era5\data\cao_tracker_result.csv"
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

if __name__ == "__main__":

    dirname = PARAMETERS["dirname"]
    filenames=os.listdir(dirname)
    dfs = list()
    for f in filenames:
        data = pd.read_csv(os.path.join(dirname,f),sep=";")
        # .stem is method for pathlib objects to get the filename w/o the extension
        data['file'] = f
        dfs.append(data)

    df = pd.concat(dfs, ignore_index=True)

    df["file"] = df["file"].str.replace(".csv", " ")
    df['time'] = pd.to_datetime(df['time'])

    df = df.dropna(subset=['lat'])
    df['ddiff'] = df['time'].diff().dt.days
    df = df.reset_index(drop=True)
    df.loc[0,'ddiff']=1

    df['group'] = (df['ddiff'] != 1).cumsum()
    df.loc[0,'group']=0

    df['group'] = df['group']+1


    group_counts = df['group'].value_counts()

    df = df[df['group'].isin(group_counts[group_counts > 1].index)]

    df = df.reset_index(drop=True)

    df['ndays']=df.groupby('group')["group"].transform('count')


    # Update group based on distance
    df['distance'] = np.nan
    for i in range(1, len(df)):
        # Calculate the distance to the previous row
        distance = haversine(df['lat'].iloc[i], df['lon'].iloc[i],
                            df['lat'].iloc[i-1], df['lon'].iloc[i-1])
        df.loc[i,'distance'] = distance
    df.loc[0,'distance']=0

    #df[df['distance']>1000]
    df['group_diff']=df['group'] - df['group'].shift()
    df['group_diff'][0]=0
    df['distance'][df['group_diff']!=0]=0


    increments = df.groupby('group')['distance'].transform(lambda x: (x >PARAMETERS["distance"] ).astype(int))
    df['increments'] = increments
    df['increments_sum'] = increments.cumsum()
    df['new_group'] = df['group'] + df['increments_sum'] 


    filtered_df = df.groupby('new_group').filter(lambda x: len(x) > 1)
    filtered_df = filtered_df.reset_index(drop=True)


    # Drop the specified columns
    filtered_df = filtered_df.drop(columns=['group', 'group_diff', 'increments', 'increments_sum'])

    # Rename the 'new_group' column to 'cao_group'
    filtered_df = filtered_df.rename(columns={'new_group': 'cao_group'})


    group = filtered_df['cao_group'] - filtered_df['cao_group'].shift()
    group[0]=0
    filtered_df['pre_group'] = group
    filtered_df['new_group'] = np.ones(np.size(filtered_df['cao_group']))
    filtered_df['new_group'][filtered_df['pre_group']==0]=0
    filtered_df['cao_group'] = filtered_df['new_group'].cumsum()


    filtered_df = filtered_df.drop(columns=['pre_group', 'new_group'])

    #lattiude filter
    group = filtered_df['lat'] - filtered_df['lat'].shift()
    group[0]=0
    filtered_df['lat_diff'] = group


    filtered_df['new_group_lat'] = np.zeros(np.size(filtered_df['lat']))
    increments = filtered_df.groupby('cao_group')['lat_diff'].transform(lambda x: (x > PARAMETERS["lat_diff_max"]).astype(int))
    filtered_df['increments_lat'] = increments
    filtered_df['increments_sum_lat'] = increments.cumsum()
    filtered_df['new_group'] = filtered_df['cao_group'] + filtered_df['increments_sum_lat'] 
    filtered_df2 = filtered_df.groupby('new_group').filter(lambda x: len(x) > 1)
    filtered_df2 = filtered_df2.reset_index(drop=True)


    group = filtered_df2['new_group'] - filtered_df2['new_group'].shift()
    group[0]=0
    filtered_df2['pre_group'] = group
    filtered_df2['new_group'] = np.ones(np.size(filtered_df2['new_group']))
    filtered_df2['new_group'][filtered_df2['pre_group']==0]=0
    filtered_df2['cao_group'] = filtered_df2['new_group'].cumsum()


    filtered_df2['ndays']=filtered_df2.groupby('cao_group')["cao_group"].transform('count')


    filtered_df2 = filtered_df2.drop(columns=['ddiff', 'lat_diff','new_group_lat','increments_lat','increments_sum_lat','new_group','pre_group'])

    filtered_df2.to_csv(PARAMETERS["output_file"],sep=';',index=False)