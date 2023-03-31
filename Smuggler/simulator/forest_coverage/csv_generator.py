#!/usr/bin/env python3
import numpy as np
from osgeo import gdal
import seaborn as sns
import matplotlib.pyplot as plt

""" This file takes the TL coverage netcdf and converts it to a csv file that can be used by the PrisonerEnv Simulator
Below is information on the TL coverage netcdf file:

    The TL coverage netcdf has a layer called "normalized_coverage". 
    It has 4 index dimensions of time,depth,latitude,longitude.  
    In the netcdf we have there is 1 time value, but 6 values for depth. 
    The "normalized_coverage" is the coverage (area that the sensor can "see") divided by the max coverage (π*max_range2).
    The CSV grid is the average over depth for each latitude/longitude pixel.  
    The values are unitless, but consider them a ratio of how well the sensor can see at each point on the map.

"""

# NETCDF_FILE = 'simulator/forest_coverage/dataset/100.0Hz_10.0m_10.0m-20.0m-50.0m-100.0m-200.0m-500.0m_90.0_CMAP.nc'
# NETCDF_FILE = 'simulator/forest_coverage/map2.nc'
NETCDF_FILE = 'map.nc'
LAYER_NAME = 'normalized_coverage'
OUTPUT_DIM_X = 2428
OUTPUT_DIM_Y = 2428
CSV_FILE = 'output.csv'

# open in GDAL
ds = gdal.Open(f"NETCDF:{NETCDF_FILE}:{LAYER_NAME}")
assert ds is not None, f"Unable to read netcdf file: {NETCDF_FILE} with layer: {LAYER_NAME}"

# transform

translated_ds = gdal.Translate('translated.nc', ds, format='NETCDF', width=OUTPUT_DIM_X, height=OUTPUT_DIM_X)
netcdf_data = translated_ds.ReadAsArray(0, 0, translated_ds.RasterXSize, translated_ds.RasterYSize)

del ds
del translated_ds

# scale to 2d
avg_data = np.mean(netcdf_data, axis=0)
avg_data = (avg_data + 0.5)/ np.max(avg_data)

min_num = np.min(avg_data)
max_num = np.max(avg_data)
target_min = 0.3
target_max = 1.0

avg_data = (avg_data - min_num) / (max_num - min_num) * (target_max - target_min) + target_min

print(np.min(avg_data), np.max(avg_data))

# print(type(avg_data))

# save csv
# np.savetxt(CSV_FILE, avg_data, delimiter=',')

np.save("map.npy", avg_data)

# sns.heatmap(avg_data)
# plt.show()

