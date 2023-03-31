""" This file generates a square map with a dense forest in the center and shallow forests everywhere else.
    This is the map we used for the December 2021 deliverable."""

import numpy as np

def generate_square_map(size_of_dense_forest, dim_x, dim_y):
    
    world_representation = np.zeros((dim_x, dim_y))
    midpoint_x = int(dim_x / 2)
    midpoint_y = int(dim_y / 2)
    world_representation[
        int(midpoint_x - size_of_dense_forest / 2):int(midpoint_x + size_of_dense_forest / 2),
        int(midpoint_y - size_of_dense_forest / 2):int(midpoint_y + size_of_dense_forest / 2)] = 0.5

    for j in range(dim_x):
        for k in range(dim_y):
            if world_representation[j, k] == 0:
                world_representation[j, k] = 1
    return world_representation

if __name__ == "__main__":
    percent_dense = 0.30
    size_of_dense_forest = int(2428 * percent_dense)
    world_representation = generate_square_map(size_of_dense_forest=size_of_dense_forest, dim_x=2428, dim_y=2428)
    # save the map as csv file
    np.savetxt("original_map.csv", world_representation, delimiter=",")
