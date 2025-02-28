# Cold air outbreak tracker
 
Python tools to track and characterize episodes of cold air outbreaks (CAO)

## Purpose
This repository 

### Repository Structure

* `data/` contains `era5_t2min_1970_2000.nc` used as input data in `test/` and `docs/`.
* `src/` stores the python functions:
    - `utils/config.py` - Set up configurations and parameters 
    - `utils/cluster.py` - Core functions for CAO identification
    - `utils/plot_utils.py` - Functions for plotting `xarray` data arrays using `matplotlib` and `cartopy`
* `test/` unit test for `src/` 
* `docs/` Jupyter notebooks tutorials