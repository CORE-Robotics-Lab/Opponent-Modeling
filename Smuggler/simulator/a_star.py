""" A star planner for the simulator. """
from html.entities import name2codepoint
import numpy as np

from .utils import distance
from fugitive_policies.a_star.utils import plot_path
from fugitive_policies.base_policy import Observation
from fugitive_policies.a_star.gridmap import OccupancyGridMap
from fugitive_policies.a_star.a_star import a_star
import copy

class AStarPlanner:
    def __init__(self, terrain,             
            max_speed=7.5,
            cost_coeff=1000,
            visualize=False):
        self.cost_coeff = cost_coeff
        self.terrain = terrain
        self.dim_x = terrain.dim_x
        self.dim_y = terrain.dim_y
        self.max_timesteps = 4320  # 72 hours = 4320 minutes
        self.first_run = True
        self.max_speed = max_speed

        self.actions = []
        self.visualize = visualize
        self.scale = (20, 20)

        # set in convert_map_for_astar()
        self.x_scale = None; self.y_scale = None

        self.x_scale = self.terrain.x_scale
        self.y_scale = self.terrain.y_scale
        data_map = self.convert_map_for_astar()
        self.gmap = OccupancyGridMap.from_terrain(data_map, 1)

    def simulate_action(self, start_location, action):
        direction = np.array([np.cos(action[1]), np.sin(action[1])])
        speed = action[0]
        new_location = start_location + direction * speed

        return new_location

    def convert_map_for_astar(self):
        """ Reduce the size of the map for the Astar algorithm - Smuggler domain"""

        data_array = np.rot90(copy.deepcopy(self.terrain.current_map), k=3)
        return data_array

    def convert_path_to_actions(self, path):
        """ Converts list of points on path to list of actions (speed, thetas)
            This function accounts for the fact that our simulator rounds actions to 
            fit on the grid map.
        """
        actions = []
        currentpos = path[0]
        for nextpos in path[1:]:
            a = self.get_actions_between_two_points(currentpos, nextpos)
            currentpos = nextpos
            actions.extend(a)
        return actions

    def get_actions_between_two_points(self, startpos, endpos):
        """ Returns list of actions (speed, thetas) to traverse between two points.
        """
        currentpos = startpos
        actions = []
        while np.array_equal(currentpos, endpos) == False:
            dist = (np.linalg.norm(np.asarray(currentpos, dtype=np.float64) - np.asarray(endpos, dtype=np.float64)))
            speed = min(dist, self.max_speed)
            theta = np.arctan2(endpos[1] - currentpos[1], endpos[0] - currentpos[0])
            action = np.array([speed, theta], dtype=np.float64)
            actions.append(action)
            currentpos = self.simulate_action(currentpos, action)

        
            # loc = self.terrain.convert_x_y_to_lat_long(currentpos)
            # print(loc)
            # Ensure that none of the path is in poor areas of the map
            # assert self.terrain.in_mountain(currentpos) == False, "Path enters mountain region"
        return actions

    def arctan_clipped(self, loc1, loc2):
        heading = np.arctan2(loc2[1] - loc1[1], loc2[0] - loc1[0])
        if heading < -np.pi:
            heading += 2 * np.pi
        elif heading > np.pi:
            heading -= 2 * np.pi
        return heading

    def scale_a_star_path(self, path, start_position, goal_location):
        """ Scale path back to original size """
        # instead of using the scaled version, use the original location for start position
        scaled_path = [start_position]
        for point in path[1:]:
            scaled_path.append((point[0] * self.x_scale, point[1] * self.y_scale))
        scaled_path.append(goal_location)
        # similarly, use the real goal location rather than the scaled version
        return scaled_path

    def reset_actions(self):
        """ Removes our A* plan """
        self.actions = []

    def predict(self, start_pos, end_pos, max_speed=7.5, movement='8N_terrain', occupancy_cost_factor=None, plot=False):
        
        if occupancy_cost_factor is None:
            occupancy_cost_factor = self.cost_coeff

        # self.reset_actions()
        self.max_speed = max_speed
        if len(self.actions) == 0:
            self.gmap.reset_visited() # clear visited flags
            # print("new prediction")
            scale_start_pos = self.terrain.convert_x_y_to_lat_long(start_pos)
            scale_start_pos = (scale_start_pos[1], scale_start_pos[0])
            scale_goal_location = self.terrain.convert_x_y_to_lat_long(end_pos)
            scale_goal_location = (scale_goal_location[1], scale_goal_location[0])
            path, path_px = a_star(scale_start_pos, scale_goal_location, self.gmap, movement=movement, occupancy_cost_factor=occupancy_cost_factor)
            
            # print(path)

            scaled_path = self.scale_a_star_path(path, start_pos, end_pos)
            self.actions = self.convert_path_to_actions(scaled_path)
            if plot:
                self.gmap.plot()
                plot_path(path_px)
        return self.actions.pop(0)