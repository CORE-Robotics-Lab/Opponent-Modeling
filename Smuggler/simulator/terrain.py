import numpy as np
import matplotlib.pyplot as plt
import colorsys
import cv2

from enum import Enum
from matplotlib.path import Path

class TerrainType(Enum):
    MOUNTAIN = 0
    DENSE_FOREST = 1
    WOODS = 2  # Shallow forest
    UNKNOWN = -1


class Terrain:
    def __init__(self,
                 forest_color_scale = 6, 
                 map_density_array = None,
                 map_density_directory = "simulator/forest_coverage/maps.npy", # directory of the map density files
                 lat_indices = (42, 84),
                 long_indices = (260, 325),
                 avoid_atlantic=True
                 ): 
        """
        Terrain instance will contain information about the terrain size and the terrain_type for each grid.
        :param dim_x: the maximum coordinate on x
        :param dim_y: the maximum coordinate on y
        :param map_density_array: a numpy array of size (n, 360, 180) denoting the latitude and longitude of every map
            Default is North is up on y-axis

        Default x, y coordinate is 

                
                |
                |
        Y Axis  |
                |
                |
                (0, 0) -----------> X axis

        """

        # Currently wave height is being represented - need to ensure that this is inverted for colors
        if map_density_array is None:
            self.map_density_array = np.load(map_density_directory)
            # import seaborn as sns
            # plt.figure()
            # sns.heatmap(self.map_density_array[0], vmin=0, vmax=20)
            # plt.show()
        else:
            raise "Not Implemented"

        self.lat_indices = lat_indices
        self.long_indices = long_indices
        self.forest_color_scale = forest_color_scale
        self.lat_dim = lat_indices[1] - lat_indices[0]
        self.long_dim = long_indices[1] - long_indices[0]

        # self.avoid_atlantic = True
        if avoid_atlantic:
            self.map_density_array[:, 56:77, 270:299] = -1
            self.map_density_array[:, 70:75, 260:270] = -1
            self.map_density_array[:, 77:82, 277:290] = -1  
            
        # block panama canal lol, boats can't go here
        self.map_density_array[:, 81, 278] = -1

        # resize map area to our desired location
        self.map = self.map_density_array[:, min(lat_indices):max(lat_indices), min(long_indices):max(long_indices)]
        midpoint_lat = (lat_indices[0] + lat_indices[1]) / 2
        # get the size of the map
        dim_x = abs(self.convert_long_to_km(midpoint_lat - 90, long_indices[1] - long_indices[0]))
        dim_y = self.convert_lat_to_km(lat_indices[1] - lat_indices[0])

        self.x_scale = dim_x / (self.long_dim) # scale km to lat/long
        self.y_scale = dim_y / (self.lat_dim) # scale km to lat/long

        self.dim_x = dim_x
        self.dim_y = dim_y


        self.current_index = 0
        self.detection_max = 15 # Scale the detection grid by dividing by this factor

        # set in self.cache terrains
        self.cached_images = []

    def visualize_map_grid(self):
        """ Visualize the grid on the lat/long grid """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import copy 

        adjusted_map = copy.deepcopy(self.map[0])
        adjusted_map[adjusted_map == -1] = 2000

        long_indices = self.long_indices
        lat_indices = self.lat_indices

        plt.figure()
        plt.imshow(adjusted_map, extent=[min(long_indices), max(long_indices), min(lat_indices), max(lat_indices)], origin='bottom')
        plt.show()

    def cache_images(self, x_dim_render, y_dim_render, num_images):
        assert num_images < self.map.shape[0]
        self.cached_images = [self.visualize(x_dim_render, y_dim_render, i, just_matrix=True) for i in range(num_images)]

    @property
    def current_map_image(self):
        return self.cached_images[self.current_index]

    def generate_random_location_open_terrain(self):
        """
        Generates a random location on the open terrain.
        :return: a list of length 2. For example, [5, 7]
        """
        x_coord = np.random.randint(0, self.dim_x - 1)
        y_coord = np.random.randint(0, self.dim_y - 1)
        while self.in_mountain([x_coord, y_coord]):
            x_coord = np.random.randint(0, self.dim_x - 1)
            y_coord = np.random.randint(0, self.dim_y - 1)

        return [x_coord, y_coord]

    @property
    def current_map(self):
        return np.flipud(self.map[self.current_index])

    def get_location_x_y(self, location):
        lat, long = self.convert_x_y_to_lat_long(location)
        return self.current_map[lat, long]

    def in_mountain(self, location):
        """ x, y location determine if in mountain """
        lat, long = self.convert_x_y_to_lat_long(location)
        # flip to index correctly corresponding to the map

        return self.current_map[lat, long] == -1

    def convert_lats_to_indices(self, lat):
        return -lat + 90

    def convert_longs_to_indices(self, long):
        return long + 180 

    def convert_lat_to_km(self, lat):
        return lat * 110.574

    def convert_long_to_km(self, lat, long):
        return long * 111.32 * np.cos(lat * np.pi / 180)

    def convert_x_y_to_lat_long(self, location):
        x = location[0]
        y = location[1]
        lat = y / self.y_scale
        long = x / self.x_scale

        if abs(lat - np.round(lat)) < 0.001:
            lat = np.round(lat)
        else:
            lat = int(lat)

        if abs(long - np.round(long)) < 0.001:
            long = np.round(long)
        else:
            long = int(long)

        # return int(np.round(lat)), int(np.round(long))
        # return int(lat), int(long)
        return int(lat), int(long)

    def out_of_bounds(self, new_location):

        lat_long = self.convert_x_y_to_lat_long(new_location)
        return lat_long[0] >= self.lat_dim or lat_long[0] <= 0 or lat_long[1] >= self.long_dim or lat_long[1] <= 0
            # new_location = np.clip(new_location, [0, 0], [self.dim_x-1, self.dim_y-1])

        # return new_location

    # def violate_edge_constraints(self, loc_x, loc_y, size_of_object_x, size_of_object_y):
    #     """
    #     Checks whether [loc_x, loc_y] exceeds the box ([0, size_of_object_x], [0, size_of_object_y])
    #     :param loc_x:
    #     :param loc_y:
    #     :param size_of_object_x:
    #     :param size_of_object_y:
    #     :return:
    #     """

    #     loc_lat, loc_long = self.convert_x_y_to_lat_long((loc_x, loc_y))
    #     if loc_long - size_of_object_x < 0:
    #         return True
    #     elif loc_long + size_of_object_x >= self.dim_x:
    #         return True
    #     elif loc_lat - size_of_object_y < 0:
    #         return True
    #     elif loc_lat + size_of_object_y >= self.dim_y:
    #         return True
    #     else:
    #         return False

    def get_color_of_pixel(self, visibility_ratio):
        """ Given the value of the visibility range from the TL coverage file, we display light green for 
        light visibility and dark green for no visibility.
        :param visibility_ratio: value between 0 and 1
        :return: rgb color value for pixel
        """
        hue = 239/360; 
        saturation = 1; 
        value = visibility_ratio / 4 + 0.2
        return colorsys.hsv_to_rgb(hue, saturation, value)

    def visualize(self, x_dim_vis, y_dim_vis, index, just_matrix=True):
        """
        Visualize the terrain.
        :param just_matrix: if True, the image will not be generated.
        :return: display_matrix ready to be shown.
        """

        # Create 2D matrix
        color_func = np.vectorize(self.get_color_of_pixel)
        resized_map = cv2.resize(self.map[index], (x_dim_vis, y_dim_vis), interpolation=cv2.INTER_NEAREST)


        # resized_map[resized_map == -1] = 0
        # Place forest on the image
        r, g, b = color_func(resized_map)
        r[resized_map == -1] = 1
        g[resized_map == -1] = 1
        b[resized_map == -1] = 1
        display_matrix = np.zeros((y_dim_vis, x_dim_vis, 3))
        display_matrix[:, :, 0] = b
        display_matrix[:, :, 1] = g
        display_matrix[:, :, 2] = r

        # print(display_matrix.meax(), display_matrix.min())
        # mountains
        # display_matrix[self.world_representation[0, :, :] == 1, :3] = 211 / 255, 211 / 255, 211 / 255

        # imshow coordinate system is different from Cartesian system. See `prisoner_env.py` for more.
        # display_matrix = np.transpose(display_materix, [1, 0, 2])
        if just_matrix:
            return np.flipud(display_matrix)

        fig, ax = plt.subplots()
        ax.imshow(np.flipud(display_matrix), origin='lower')
        plt.show()
        return display_matrix

    def terrain_for_range(self, range_x, range_y):
        raise "Not Implemented"
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
        lat, long = self.convert_x_y_to_lat_long(location)

        detect = self.current_map[lat, long]
        if detect < 0:
            return 0
        return detect / self.detection_max

    def terrain_given_location(self, location):
        """
        provides terrain_type for a location
        :param location: a list of length 2. For example, [5, 7]
        :return: the terrain_type of the given location according to the terrain.
        """
        raise NotImplementedError
        location = np.clip(location, 0, 2427)
        if self.world_representation[0, location[0], location[1]] == 1:
            return TerrainType.MOUNTAIN
        elif self.world_representation[1, location[0], location[1]] == 1:
            return TerrainType.DENSE_FOREST
        elif self.world_representation[2, location[0], location[1]] == 1:
            return TerrainType.WOODS
        else:
            return TerrainType.UNKNOWN




if __name__ == "__main__":
    # test code
    terrain = Terrain(lat_indices = (70, 102),
                 long_indices = (215, 286))

    import seaborn as sns
    import matplotlib.pyplot as plt
    import copy 

    adjusted_map = copy.deepcopy(terrain.current_map)
    adjusted_map[adjusted_map == -1] = 2000

    plt.figure()
    sns.heatmap(adjusted_map, vmin=0, vmax=20)
    plt.show()

    # location = (0, 4400)

    ratio_render = 0.1
    x_dim_render = int(terrain.dim_x * ratio_render)
    y_dim_render = int(terrain.dim_y * ratio_render)

    m = terrain.visualize(x_dim_vis=x_dim_render, y_dim_vis=y_dim_render, index=0, just_matrix=False)
    plt.figure()
    plt.imshow(np.flipud(m), origin='lower')
    plt.show()



    # a = terrain.convert_x_y_to_lat_long((1489.0838755, 221.148))
    # b = terrain.convert_x_y_to_lat_long((1484.5909561014512, 227.153303999))
    # print(a, b)
    # a = terrain.in_mountain((1489.0838755, 221.148))
    # b = terrain.in_mountain((1484.5909561014512, 227.153303999))
    # print(a, b)