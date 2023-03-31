from osgeo import gdal
import numpy as np
import os
import glob

def read_map_file(map_path, target_min=0.3, target_max=1.0):
    """ Reads the .nc map file and returns a numpy array of the data """
    OUTPUT_DIM_X = 2428
    OUTPUT_DIM_Y = 2428
    LAYER_NAME = 'normalized_coverage'
    # open in GDAL
    ds = gdal.Open(f"NETCDF:{map_path}:{LAYER_NAME}")
    assert ds is not None, f"Unable to read netcdf file: {map_path} with layer: {LAYER_NAME}"

    # transform
    translated_ds = gdal.Translate(map_path, ds, format='NETCDF', width=OUTPUT_DIM_X, height=OUTPUT_DIM_Y)
    netcdf_data = translated_ds.ReadAsArray(0, 0, translated_ds.RasterXSize, translated_ds.RasterYSize)

    # scale to 2d
    avg_data = np.mean(netcdf_data, axis=0)

    min_num = np.min(avg_data)
    max_num = np.max(avg_data)
    avg_data = (avg_data - min_num) / (max_num - min_num) * (target_max - target_min) + target_min
    return avg_data

def convert_multiple_maps(dataset_dir, target_min=0.3, target_max=1.0):
    """ Converts all the .nc files in the directory to .npy files """
    root_dir = "maps"
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    file_paths = glob.glob(dataset_dir + '/**/*.nc', recursive=True)
    print(file_paths)
    print(len(file_paths))
    for i, file_path in enumerate(file_paths):
        avg_data = read_map_file(file_path, target_min, target_max)
        np.save(os.path.join(root_dir, f"{i}.npy"), avg_data)

if __name__ == "__main__":
    convert_multiple_maps("/home/sean/Desktop/forest_coverage_maps/HYCOM/MAR_full")
