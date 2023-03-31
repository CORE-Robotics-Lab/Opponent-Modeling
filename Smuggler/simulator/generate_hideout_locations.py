import sys, os
sys.path.append(os.getcwd())

import numpy as np

from simulator.terrain import Terrain
from simulator.forest_coverage.generate_square_map import generate_square_map



def dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def produce_hideout_distribution(dim_x, dim_y, num_hideouts, terrain, min_distance=400):
    """
    Produce a set of hideouts that are a certain distance from each other
    
    Args:
        dim_x: x dimension of the map
        dim_y: y dimension of the map
    """

    hideouts = set()
    while len(hideouts) < num_hideouts:
        x = np.random.randint(0, dim_x)
        y = np.random.randint(0, dim_y)

        hideout_location = (x, y)
        if terrain:
            if terrain.location_in_mountain(hideout_location):
                continue
        
        if len(hideouts) > 0:
            dist_hideouts = min([dist(hideout, hideout_location) for hideout in hideouts])
            if dist_hideouts < min_distance:
                continue
        hideouts.add(hideout_location)

    return list(hideouts)

def write_cameras_to_file(camera_list, file_path):
    """
    Write camera locations to a file
    """
    with open(file_path, 'w') as f:
        for camera in camera_list:
            f.write(f'u,{camera[0]},{camera[1]}\n')

if __name__ == "__main__":
    # use original map with size 2428x2428
    dim_x = 2428
    dim_y = 2428
    percent_dense = 0.3
    size_of_dense_forest = int(dim_x * percent_dense)
    forest_density_array = generate_square_map(size_of_dense_forest=size_of_dense_forest, dim_x=dim_x, dim_y=dim_y)
    terrain = Terrain(dim_x=dim_x, dim_y=dim_y, forest_color_scale = 1, forest_density_array = forest_density_array)

    # h = produce_hideout_distribution(dim_x=dim_x, dim_y=dim_y, num_hideouts=20, terrain=terrain)

    h = [(2077, 2151), (234, 2082), (2170, 603), (1191, 950), (37, 1293), (563, 750), (1890, 30), (2314, 86), (1151, 2369), (1119, 1623), (356, 78), (1636, 2136), (1751, 1433), (602, 1781), (1638, 1028), (2276, 1007), (1482, 387), (980, 118), (457, 1221), (2258, 1598)]

    known = h[::2] # [(2077, 2151), (2170, 603), (37, 1293), (1890, 30), (1151, 2369), (356, 78), (1751, 1433), (1638, 1028), (1482, 387), (457, 1221)]
    unknown = h[1::2] # [(234, 2082), (1191, 950), (563, 750), (2314, 86), (1119, 1623), (1636, 2136), (602, 1781), (2276, 1007), (980, 118), (2258, 1598)]

    print(known)
    print(unknown)

    import matplotlib.pyplot as plt
    plt.scatter(np.array(known)[:, 0], np.array(known)[:, 1], color='red')
    plt.scatter(np.array(unknown)[:, 0], np.array(unknown)[:, 1], color='blue')
    plt.show()