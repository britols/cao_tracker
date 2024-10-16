# cao_tracker
Detection and tracking of Cold Air Outbreaks (CAO) in North America

```
["dayofyear", "weekofyear", "month", "season"] + ["skill"] 
["dayofyear", "weekofyear", "month", "season","skill"]  
```
```
OPTIONS = {
    "a": "aaaa",
    "b": "bbbb"
}
new_options{
    "b": "ddddd"
}

OPTIONS.update(new_options)
OPTIONS
{'a': 'aaaa', 'b': 'ddddd'}
```

Cao Results

82 Winters (1941 to 2022)
First CAO: 12/02/1940
Last CAO: 02/21/2022
N CAO: 890


TO DO: Consider the CAO size before filtering by latitude, sometimes the latitude increases because the tracker is considering more points inside the CAO

TO DO:

Boxplot per decade
Correlations (Mean Anomaly, Size, Duration)