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

import seaborn as sns

if __name__ == '__main__':


    terrain_path = "simulator/forest_coverage/map_set/0.npy"
    terrain_map = np.load(terrain_path)
    terrain = Terrain(forest_density_array = terrain_map)

    mountains = terrain.world_representation[0, :, :]
    terrain_map = terrain.world_representation[1, :, :]

    print(mountains)

    mask = np.where(mountains == 1)
    terrain_map[mask] = -1

    # scale = 1/20
    # size = (int(terrain_map.shape[0]*scale), int(terrain_map.shape[1]*scale))
    # resized = cv2.resize(terrain_map, (2048, 2048), interpolation=cv2.INTER_NEAREST)

    scale = (20, 20)

    x_remainder = terrain_map.shape[0] % scale[0]
    y_remainder = terrain_map.shape[1] % scale[1]

    pooled = skimage.measure.block_reduce(terrain_map, (scale[0], scale[1]), np.min)

    pooled = pooled[:-x_remainder, :-y_remainder]

    pooled = np.flipud(np.rot90(pooled, k=1))
    print(pooled.shape)

    # print(resized.shape)

    # sns.heatmap(np.rot90(pooled, k=1))
    # plt.show()

    # load the map
    gmap = OccupancyGridMap.from_terrain(pooled, 1)

    # set a start and an end node (in meters)
    # start_node = (0, 80)
    # goal_node = (60, 40)

    start_node = (113, 113)
    goal_node = (0, 100)

    # run A*
    path, path_px = a_star(start_node, goal_node, gmap, movement='8N')
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

    # plt.show()
    plt.savefig('a_star_8n.png')
