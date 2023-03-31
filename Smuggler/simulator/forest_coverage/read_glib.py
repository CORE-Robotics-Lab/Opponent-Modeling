import pygrib
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt 

def get_map_files(path):
    """
    Get all files with "sig_wav_ht" in the name in a path.

    Args:
        path: path to the directory containing the files.
    """
    files = []
    for file in sorted(os.listdir(path)):
        if file.find("sig_wav_ht") != -1:
            files.append(os.path.join(path, file))
    return files

def get_map_data(file):
    """
    Get the map data from file

    latitude is y-axis
    longitude is x-axis

    """

    # lon_min += 90
    # lon_max += 90
    grbs = pygrib.open(file)
    grb = grbs.select(name = 'Signific.height,combined wind waves+swell')[0]
    vals = np.flipud(grb.values)
    return vals

def get_all_files(path):
    """
    Get all files in the directory and turn into numpy array.
    """
    files = get_map_files(path)
    maps = []
    maxes = []
    for file in files:
        map = get_map_data(file)
        maps.append(map)

    maps = np.array(maps)
    # make the maps have the smallest shape
    maps[maps==9999] = -1
    return maps
    
def resize_maps():
    pass

if __name__ == "__main__":

    maps = get_all_files("simulator/wave_data")
    # 
    print(maps.shape)
    
    print(np.max(maps), np.min(maps))

    np.save("simulator/forest_coverage/maps", maps)

    # files = get_map_files("simulator/wave_data")
    # print(files)

    # for file in files:
    #     timestamp = file.split("_")[2][:4]

    # # print(len(files))
    # m = get_map_data(files[0], lon_min=265, lon_max=305, lat_min=36, lat_max=103
    # print(m.shape)
    # # print(m.shape)
    plt.figure()
    sns.heatmap(maps[0], vmin=0, vmax=20)
    plt.show()