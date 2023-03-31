import numpy as np

from .utils import clip_theta, distance, c_str
import matplotlib.pyplot as plt
import time
from fugitive_policies.custom_queue import QueueFIFO

DIM_X = 2428
DIM_Y = 2428

MOUNTAIN_OUTER_RANGE = 150
MOUNTAIN_INNER_RANGE = 155
import math

import skimage.measure
from fugitive_policies.base_policy import Observation
from fugitive_policies.a_star.gridmap import OccupancyGridMap
from fugitive_policies.a_star.a_star import a_star
import copy

class AStarPolicy:
    def __init__(self, env,             
            max_speed=7.5,
            cost_coeff=1000,
            visualize=False):
        self.env = env
        self.cost_coeff = cost_coeff
        self.terrain = self.env.terrain
        self.dim_x = env.terrain.dim_x
        self.dim_y = env.terrain.dim_y
        self.num_known_cameras = env.num_known_cameras
        self.num_search_parties = env.num_search_parties
        self.num_helicopters = env.num_helicopters
        self.num_known_hideouts = env.num_known_hideouts
        self.num_unknown_hideouts = env.num_unknown_hideouts
        self.max_timesteps = 4320  # 72 hours = 4320 minutes
        self.observations = Observation(self.terrain, self.num_known_cameras, self.num_helicopters, self.num_known_hideouts, self.num_unknown_hideouts, self.num_search_parties)
        self.first_run = True
        self.max_speed = max_speed

        self.actions = []
        self.visualize = visualize
        self.scale = (20, 20)

        # set in convert_map_for_astar()
        self.x_scale = None; self.y_scale = None
        pooled = self.convert_map_for_astar()
        self.gmap = OccupancyGridMap.from_terrain(pooled, 1)


    def get_closest_hideout(self, location, hideout_list):
        min_dist = np.inf
        closest_hideout = None
        for hideout in hideout_list:
            dist = distance(location, hideout.location)
            if dist < min_dist:
                min_dist = dist
                closest_hideout = hideout
        return closest_hideout, min_dist

    def get_closest_hideouts(self, location):
        closest_known_hideout, _ = self.get_closest_hideout(location, self.observations.known_hideout_list)
        closest_unknown_hideout, _ = self.get_closest_hideout(location, self.observations.unknown_hideout_list)

        return closest_known_hideout, closest_unknown_hideout

    def simulate_action(self, start_location, action):
        direction = np.array([np.cos(action[1]), np.sin(action[1])])
        speed = action[0]
        new_location = np.round(start_location + direction * speed)
        new_location[0] = np.clip(new_location[0], 0, self.dim_x - 1)
        new_location[1] = np.clip(new_location[1], 0, self.dim_y - 1)
        new_location = new_location.astype(np.int)
        return new_location

    def convert_map_for_astar(self):
        """ Reduce the size of the map for the Astar algorithm """
        mountains = copy.deepcopy(self.terrain.world_representation[0, :, :])
        terrain_map = copy.deepcopy(self.terrain.world_representation[1, :, :])

        mask = np.where(mountains == 1)
        terrain_map[mask] = -1

        x_remainder = terrain_map.shape[0] % self.scale[0]
        y_remainder = terrain_map.shape[1] % self.scale[1]
        pooled = skimage.measure.block_reduce(terrain_map, (self.scale[0], self.scale[1]), np.min)
        pooled = pooled[:-x_remainder, :-y_remainder]
        pooled = np.flipud(np.rot90(pooled, k=1))

        self.x_scale = self.terrain.dim_x / pooled.shape[0]
        self.y_scale = self.terrain.dim_y / pooled.shape[1]

        return pooled

    def plan(self, endpos):
        pass

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
            This function accounts for the fact that our simulator rounds actions to 
            fit on the grid map.
        """
        currentpos = startpos
        actions = []
        while np.array_equal(currentpos, endpos) == False:
            dist = (np.linalg.norm(np.asarray(currentpos) - np.asarray(endpos)))
            speed = min(dist, self.max_speed)
            theta = np.arctan2(endpos[1] - currentpos[1], endpos[0] - currentpos[0])
            action = np.array([speed, theta], dtype=np.float32)
            actions.append(action)
            currentpos = self.simulate_action(currentpos, action)

            # if self.terrain.world_representation[0, currentpos[0], currentpos[1]] == False:
                # print("In mountain!!")

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
            scaled_path.append((int(point[0] * self.x_scale), int(point[1] * self.y_scale)))
        scaled_path.append(goal_location)
        # similarly, use the real goal location rather than the scaled version
        return scaled_path

    def predict(self, observation, goal='closest', deterministic=True, plot=False):
        self.observations.process_observation(observation)
        if len(self.actions) == 0:
            closest_known_hideout, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
            goal_location = closest_unknown_hideout.location
            
            start_pos = list(map(int, self.observations.location))
            scale_start_pos = (int(start_pos[0] / self.x_scale), int(start_pos[1] / self.y_scale))
            scale_goal_location = (int(goal_location[0] / self.x_scale), int(goal_location[1] / self.y_scale))
            path, path_px = a_star(scale_start_pos, scale_goal_location, self.gmap, movement='8N', occupancy_cost_factor=self.cost_coeff)
            scaled_path = self.scale_a_star_path(path, start_pos, goal_location)
            self.actions = self.convert_path_to_actions(scaled_path)
        return [self.actions.pop(0)]