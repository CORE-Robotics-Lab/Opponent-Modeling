import numpy as np
import matplotlib.pyplot as plt
import colorsys

from enum import Enum
from matplotlib.path import Path

class TerrainType(Enum):
    MOUNTAIN = 0
    DENSE_FOREST = 1
    WOODS = 2  # Shallow forest
    UNKNOWN = -1


class Terrain:
    def __init__(self,
                 dim_x=2428,
                 dim_y=2428,
                #  num_mountains=2,
                 percent_mountain=.08,
                 percent_dense=.30,
                 forest_color_scale = 6, 
                 forest_density_array = None,
                 mountain_locations = [(400, 300), (1600, 1800)]):
        """
        Terrain instance will contain information about the terrain size and the terrain_type for each grid.
        :param dim_x: the maximum coordinate on x
        :param dim_y: the maximum coordinate on y
        :param percent_mountain: The amount of area each mountain should assume (in comparison to total map area)
        :param percent_dense: The amount of area dense forest should assume (in comparison to total map area)
        :param forest_color_scale: Scale for how the forest color is determined, does not affect actual forest density
        :param mountain_locations: List of (x, y) coordinates for mountain centers
        """
        self.num_mountains = len(mountain_locations)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.percent_dense = percent_dense
        self.percent_mountain = percent_mountain
        self.forest_color_scale = forest_color_scale
        self.forest_density_array = forest_density_array

        # compute area of different terrain objects
        self.size_of_mountain = int(self.dim_x * self.percent_mountain)

        # mountains, dense, shallow
        self.obj_to_dim = {'mountains': 0, 'forest': 1}
        self.num_objects = len(self.obj_to_dim)
        self.world_representation = np.zeros((self.num_objects, self.dim_x, self.dim_y))

        self.mountain_locations = []

        # populate world with terrain, functions update world_representation
        if self.num_mountains > 0:
            x_mountains = [loc[0] for loc in mountain_locations]
            y_mountains = [loc[1] for loc in mountain_locations]
            self.place_mountains(x_mountains=x_mountains, y_mountains=y_mountains)
        self.place_forests()
        # self.visualize()

    def place_forests(self):
        """ Based on TL coverage data, place forests on the grid"""
        assert self.forest_density_array.shape == (self.dim_x, self.dim_y)
        forest_dimension = self.obj_to_dim['forest']
        self.world_representation[forest_dimension, :, :] = self.forest_density_array
        

    def place_mountains(self, x_mountains=None, y_mountains=None):
        """
        This function places the number of mountain. Each mountain centerpoint is randomly initialized,
        ensuring that the mountain is placed far away from other mountains and does not go outside the
        map dimensions. Mountains are initialized as squares with area equal to percent_mountain
        """

        if x_mountains is None:
            random_placement = True
        else:
            random_placement = False

        mountain_locations = []
        for i in range(self.num_mountains):
            if random_placement:
                x = np.random.randint(self.dim_x)
                y = np.random.randint(self.dim_y)
            else:
                x = x_mountains[i]
                y = y_mountains[i]
            # make sure object is a certain distance from other objectives

            # ensure we didn't violate edge constraints
            while (i > 0 and np.linalg.norm(np.array([x, y]) - mountain_locations[-1]) < 300) or \
                    self.violate_edge_constraints(x, y, self.size_of_mountain / 2, self.size_of_mountain / 2):
                x = np.random.randint(self.dim_x)
                y = np.random.randint(self.dim_y)

            # append (y, x) to keep consistent with defaults that we had before
            # However, we're passing in (x, y) with (0,0) in the bottom left and (2428, 2428) in the top right
            mountain_locations.append((y, x))

        self.mountain_locations = mountain_locations
        
        for i in range(len(mountain_locations)):
            # code for square mountains
            # self.world_representation[0,
            # int(mountain_locations[i][0] - self.size_of_mountain / 2):int(
            #     mountain_locations[i][0] + self.size_of_mountain / 2),
            # int(mountain_locations[i][1] - self.size_of_mountain / 2):int(
            #     mountain_locations[i][1] + self.size_of_mountain / 2)] = 1

            mask3 = self.create_complex_mountain_shape(mountain_locations[i], visualize=False)
           
            # update representation with mask
            self.world_representation[0,:,:] = np.logical_or(self.world_representation[0,:,:], mask3)



    def create_complex_mountain_shape(self, center, visualize=False):
        # vertices of polygon (rectangle)
        xc = np.array([int(center[0] - self.size_of_mountain / 2),
                       int(center[0] - self.size_of_mountain / 2),
                       int(center[0] + self.size_of_mountain / 2),
                       int(center[0] + self.size_of_mountain / 2),
                       ])
        yc = np.array([int(center[1] + self.size_of_mountain / 4),
                       int(center[1] - self.size_of_mountain / 4),
                       int(center[1] - self.size_of_mountain / 4),
                       int(center[1] + self.size_of_mountain / 4),
                       ])
        xycrop = np.vstack((xc, yc)).T

        # xy coordinates for each pixel in the image
        nr, nc = self.world_representation[0].shape
        ygrid, xgrid = np.mgrid[:nr, :nc]
        xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T

        # construct a Path from the vertices
        pth = Path(xycrop, closed=False)

        # test which pixels fall within the path
        mask = pth.contains_points(xypix)

        # reshape to the same size as the image
        mask = mask.reshape(self.world_representation[0].shape)

        # adding to complexity of shape
        # vertices of polygon (rectangle)
        angle_of_rotation = np.pi / 2
        width = self.size_of_mountain
        height = self.size_of_mountain / 2

        # order - top-left, bottom-left, bottom-right, top-right
        x_rotated = np.array([int(center[0] - (width / 2 * np.cos(angle_of_rotation)) - (
                    height / 2 * np.sin(angle_of_rotation))),
                              int(center[0] - (width / 2 * np.cos(angle_of_rotation)) + (
                                          height / 2 * np.sin(angle_of_rotation))),
                              int(center[0] + (width / 2 * np.cos(angle_of_rotation)) + (
                                          height / 2 * np.sin(angle_of_rotation))),
                              int(center[0] + (width / 2 * np.cos(angle_of_rotation)) - (
                                          height / 2 * np.sin(angle_of_rotation))),
                              ])
        y_rotated = np.array([int(center[1] - (width / 2 * np.sin(angle_of_rotation)) + (
                    height / 2 * np.cos(angle_of_rotation))),
                              int(center[1] - (width / 2 * np.sin(angle_of_rotation)) - (
                                          height / 2 * np.cos(angle_of_rotation))),
                              int(center[1] + (width / 2 * np.sin(angle_of_rotation)) - (
                                          height / 2 * np.cos(angle_of_rotation))),
                              int(center[1] + (width / 2 * np.sin(angle_of_rotation)) + (
                                          height / 2 * np.cos(angle_of_rotation))),
                              ])

        xycrop = np.vstack((x_rotated, y_rotated)).T

        # xy coordinates for each pixel in the image
        nr, nc = self.world_representation[0].shape
        ygrid, xgrid = np.mgrid[:nr, :nc]
        xypix = np.vstack((xgrid.ravel(), ygrid.ravel())).T

        # construct a Path from the vertices
        pth = Path(xycrop, closed=False)

        # test which pixels fall within the path
        mask2 = pth.contains_points(xypix)

        # reshape to the same size as the image
        mask2 = mask2.reshape(self.world_representation[0].shape)

        mask3 = np.logical_or(mask, mask2)
        if visualize:
            fig, ax = plt.subplots(1, 3)
            ax[0].imshow(mask, cmap=plt.cm.gray)
            ax[0].set_title('mask')
            ax[1].imshow(mask2, cmap=plt.cm.gray)
            ax[1].set_title('mask')
            ax[2].imshow(mask3, cmap=plt.cm.gray)
            ax[2].set_title('mask')
            plt.show()
        return mask3

    def violate_edge_constraints(self, loc_x, loc_y, size_of_object_x, size_of_object_y):
        """
        Checks whether [loc_x, loc_y] exceeds the box ([0, size_of_object_x], [0, size_of_object_y])
        :param loc_x:
        :param loc_y:
        :param size_of_object_x:
        :param size_of_object_y:
        :return:
        """
        if loc_x - size_of_object_x < 0:
            return True
        elif loc_x + size_of_object_x >= self.dim_x:
            return True
        elif loc_y - size_of_object_y < 0:
            return True
        elif loc_y + size_of_object_y >= self.dim_y:
            return True
        else:
            return False

    def get_color_of_pixel(self, visibility_ratio):
        """ Given the value of the visibility range from the TL coverage file, we display light green for 
        light visibility and dark green for no visibility.
        :param visibility_ratio: value between 0 and 1
        :return: rgb color value for pixel
        """
        hue = 113/360
        saturation = 1
        value = visibility_ratio * self.forest_color_scale
        return colorsys.hsv_to_rgb(hue, saturation, value)


    def visualize(self, just_matrix=False):
        """
        Visualize the terrain.
        :param just_matrix: if True, the image will not be generated.
        :return: display_matrix ready to be shown.
        """
        # Create 2D matrix
        color_func = np.vectorize(self.get_color_of_pixel)

        # Place forest on the image
        r, g, b = color_func(self.world_representation[1])
        display_matrix = np.zeros((self.dim_x, self.dim_y, 3))
        display_matrix[:, :, 0] = r
        display_matrix[:, :, 1] = g
        display_matrix[:, :, 2] = b

        # mountains
        display_matrix[self.world_representation[0, :, :] == 1, :3] = 211 / 255, 211 / 255, 211 / 255

        # imshow coordinate system is different from Cartesian system. See `prisoner_env.py` for more.
        display_matrix = np.transpose(display_matrix, [1, 0, 2])
        if just_matrix:
            return display_matrix

        fig, ax = plt.subplots()
        ax.imshow(display_matrix, origin='lower')
        plt.show()
        return display_matrix

    def terrain_for_range(self, range_x, range_y):
        """
        provides local terrain information
        :param range_x: a list of length 2 ([a ,b]) indicating range a to b (inclusive)
        :param range_y: a list of length 2 ([a ,b]) indicating range a to b (inclusive)
        :return: an ndarray of terrain information of the requested range
        """
        local_terrain = np.zeros(shape=[range_x[1] - range_x[0] + 1, range_y[1] - range_y[0] + 1])
        for x in range(range_x[0], range_x[1] + 1):
            for y in range(range_y[0], range_y[1] + 1):
                if self.violate_edge_constraints(x, y, self.dim_x, self.dim_y):
                    local_terrain[x - range_x[0], y - range_y[0]] = -1
                else:
                    local_terrain[x - range_x[0], y - range_y[0]] = self.terrain_given_location([x, y]).value
        return local_terrain

    def detection_coefficient_given_location(self, location):
        x_clipped = np.clip(location[0], 0, self.dim_x - 1)
        y_clipped = np.clip(location[1], 0, self.dim_y - 1)
        location = np.array([x_clipped, y_clipped])
        forest_dimension = self.obj_to_dim['forest']
        return self.world_representation[forest_dimension, location[0], location[1]]

    def terrain_given_location(self, location):
        """
        provides terrain_type for a location
        :param location: a list of length 2. For example, [5, 7]
        :return: the terrain_type of the given location according to the terrain.
        """
        location = np.clip(location, 0, 2427)
        if self.world_representation[0, location[0], location[1]] == 1:
            return TerrainType.MOUNTAIN
        elif self.world_representation[1, location[0], location[1]] == 1:
            return TerrainType.DENSE_FOREST
        elif self.world_representation[2, location[0], location[1]] == 1:
            return TerrainType.WOODS
        else:
            return TerrainType.UNKNOWN
            
    def location_in_mountain(self, location):
        """
        checks whether a location is in a mountain
        :param location: a list of length 2. For example, [5, 7]
        :return: True if the location is in a mountain, False otherwise.
        """
        return self.world_representation[0, location[0], location[1]] == 1

if __name__ == "__main__":
    # test code
    terrain = Terrain()
