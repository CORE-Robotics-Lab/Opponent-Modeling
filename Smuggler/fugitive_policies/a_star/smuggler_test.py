import os, sys
sys.path.append(os.getcwd())

import numpy as np
from fugitive_policies.a_star.gridmap import OccupancyGridMap
import matplotlib.pyplot as plt
from fugitive_policies.a_star.a_star import a_star
from fugitive_policies.a_star.utils import plot_path

from simulator.terrain import Terrain
import cv2
import skimage.measure
import copy

import seaborn as sns

if __name__ == '__main__':
    terrain = Terrain( 
                lat_indices = (55, 100),
                long_indices = (250, 325))
    m = copy.deepcopy(np.rot90(terrain.current_map, k=3))

    sns.heatmap(m)
    plt.show()
    print(m.shape)

    # load the map
    gmap = OccupancyGridMap.from_terrain(m, 1)

    start_node = (0, 0)
    goal_node = (25, 25)

    # run A*
    path, path_px = a_star(start_node, goal_node, gmap, movement='8N')

    print(path_px)

    gmap.plot()

    if path:
        # plot resulting path in pixels over the map
        plot_path(path_px)
    else:
        print('Goal is not reachable')

        # plot start and goal points over the map (in pixels)
        start_node_px = gmap.get_index_from_coordinates(start_node[0], start_node[1])
        goal_node_px = gmap.get_index_from_coordinates(goal_node[0], goal_node[1])

        plt.plot(start_node_px[0], start_node_px[1], 'ro')
        plt.plot(goal_node_px[0], goal_node_px[1], 'go')

    plt.show()