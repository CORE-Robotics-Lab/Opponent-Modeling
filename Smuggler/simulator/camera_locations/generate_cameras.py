""" Generate a list of cameras to cover n percentage of the map, continuously sample"""

import numpy as np

from simulator.terrain import Terrain
from simulator.forest_coverage.generate_square_map import generate_square_map

def dist(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def produce_camera_distribution(dim_x, dim_y, camera_range, target_camera_density, terrain=None):
    """
    Produce a camera distribution that a certain percentage of the map
    
    Args:
        dim_x: x dimension of the map
        dim_y: y dimension of the map
        camera_range: range of the camera
        target_camera_density: percentage of the map covered by cameras
    """

    unknown_hideouts = [[376, 1190], [909, 510], [397, 798], [2059, 541], [2011, 103], [901, 883], [1077, 1445], [602, 372], [80, 2274], [279, 477]]
    camera_density = 0
    camera_set = set()
    while camera_density < target_camera_density:
        x = np.random.randint(0, dim_x)
        y = np.random.randint(0, dim_y)

        camera_location = (x, y)
        if terrain:
            if terrain.location_in_mountain(camera_location):
                continue
        
        in_range_check = sum([dist(camera_location, camera) < camera_range for camera in camera_set])
        dist_hideouts = sum([dist(hideout, camera_location) < camera_range for hideout in unknown_hideouts])
        if camera_location not in camera_set and in_range_check == 0 and dist_hideouts == 0:
            camera_set.add(camera_location)

        camera_density = (len(camera_set) * np.pi * camera_range**2) / (dim_x * dim_y)

    return list(camera_set)

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
    terrain = Terrain(dim_x=dim_x, dim_y=dim_y, forest_color_scale = 1, forest_density_array = forest_density_array, place_mountains=True)

    for percentage in range(40, 90, 10):
        print(percentage/100)
        file_path = f"simulator/camera_locations/{percentage}_percent_cameras.txt"
        cameras = produce_camera_distribution(2428, 2428, 100, percentage/100, terrain)
        write_cameras_to_file(cameras, file_path)

        known_camera_locations = [[1000, 800], [1400, 400], [1750, 1800], [1580, 1200], [2200, 2200]]
        with open(file_path, 'a') as f:
            for camera in known_camera_locations:
                f.write(f'k,{camera[0]},{camera[1]}\n')

    # lines = open("simulator/camera_locations/original.txt", "r").readlines()
    # cameras = [list(map(int, i.strip().split(","))) for i in lines]
    # print(cameras)