import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def set_white_color(original_cmap):
    # Get the colors from the original colormap
    clrs = original_cmap(np.linspace(0, 1, original_cmap.N))
    # Set the first color to white
    clrs[0, :] = [1, 1, 1, 1] #RGBA values for white
    # Create a new colormap
    new_cmap = colors.ListedColormap(clrs)
    return(new_cmap)

#def da_plot(da,proj_from=ccrs.PlateCarree(),cmap='viridis',type=):
#    da.plot(transform=proj_from,cmap=cmap)


def da_plot(da, proj_from=ccrs.PlateCarree(),cmap='viridis', type="plot"):
    if type == "contour":
        da.plot.contour(transform=proj_from,colors='yellow',linewidths=0.5)
    else:
        da.plot(transform=proj_from,cmap=cmap)
    
def da_plot_zero_centered(da,proj_from=ccrs.PlateCarree(),
                z = [-3,-2,-1.5,-1,-0.5,0.5,1,1.5,2,3],
                cmap_stdev = colors.LinearSegmentedColormap.from_list("", ["purple","darkblue","blue","lightblue","white","lightcoral","red","darkred","pink"])):
    
    norm = colors.TwoSlopeNorm(vmin=np.min(z), vcenter=0, vmax=np.max(z))
    da.plot(transform=proj_from,cmap=cmap_stdev,norm=norm,levels=z,extend="both")

def map_plot(proj_from=ccrs.PlateCarree(),proj_to=ccrs.NorthPolarStereo(central_longitude=-100),extent=True):
    axes = plt.axes(projection=proj_to)
    axes.coastlines()
    axes.add_feature(cfeature.STATES, edgecolor='black', linewidth=0.5)
    if(extent):
        axes.set_extent((-140, -50, 20, 90),proj_from)

def convert_cmass(cm_lat,cm_lon,latitude,longitude,proj_to=ccrs.NorthPolarStereo(central_longitude=-100),proj_from=ccrs.PlateCarree()):
    if any(x > len(longitude) for x in cm_lon):
        print("invalid cm coords")
        return cm_lat,cm_lon
    if any(y > len(latitude) for y in cm_lat):
        print("invalid cm coords")
        return cm_lat,cm_lon
    
    cm_lat = latitude[cm_lat.astype(int)]
    cm_lon = longitude[cm_lon.astype(int)]
    new_coords = proj_to.transform_points(proj_from, cm_lon, cm_lat)
    return(new_coords)


