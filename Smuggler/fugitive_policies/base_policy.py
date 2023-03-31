import numpy as np

from .utils import clip_theta, distance, c_str
import matplotlib.pyplot as plt
import time
from fugitive_policies.custom_queue import QueueFIFO

MOUNTAIN_OUTER_RANGE = 150
MOUNTAIN_INNER_RANGE = 155
import math


class Observation:
    """ Observation data 
    0: timestep
    n known cameras (location[0], location[1]) scaled by size of board
    fugitive location (x, y) scaled by size of board
    fugitive velocity (x, y) scaled by 15 (max speed) and np.pi for heading
    fugitive detection of parties
    local terrain (need full terrain so pass in env.terrain)
    """
    def __init__(self, terrain, num_known_cameras, num_helicopters, num_known_hideouts, num_unknown_hideouts, num_search_parties, num_rendezvous_points):
        self.terrain = terrain
        self.num_known_cameras = num_known_cameras
        self.num_helicopters = num_helicopters
        self.num_known_hideouts = num_known_hideouts
        self.num_unknown_hideouts = num_unknown_hideouts
        self.num_search_parties = num_search_parties
        self.num_rendezvous_points = num_rendezvous_points
        
        self.location = np.zeros(2)
        self.goal_location = np.zeros(2) 
        self.camera_list = []
        self.heli_list = []
        self.unknown_hideout_list = []
        self.known_hideout_list = []
        self.search_party_list = []
        
        DetectionObject.DIM_X = self.terrain.dim_x
        DetectionObject.DIM_Y = self.terrain.dim_y

        Hideout.DIM_X = self.terrain.dim_x
        Hideout.DIM_Y = self.terrain.dim_y

        RendezvousPoint.DIM_X = self.terrain.dim_x
        RendezvousPoint.DIM_Y = self.terrain.dim_y
        

    def reset(self):
        self.location = np.zeros(2)
        self.camera_list = []
        self.heli_list = []
        self.unknown_hideout_list = []
        self.known_hideout_list = []
        self.search_party_list = []

    def process_observation(self, observations):
        self.reset()
        self.timestep = observations[0]
        start = 1
        for _ in range(self.num_known_cameras):
            camera = Camera(observations[start:start+2])
            self.camera_list.append(camera)
            start += 2

        for _ in range(self.num_known_hideouts + self.num_unknown_hideouts):
            hideout = Hideout(observations[start], observations[start+1:start+3])
            if observations[start] == 1:
                self.known_hideout_list.append(hideout)
            else:
                self.unknown_hideout_list.append(hideout)
            start += 3

        self.location = np.rint(observations[start:start+2] * np.array([self.terrain.dim_x, self.terrain.dim_y]))
        start += 2

        self.action = np.array([observations[start] * 15, observations[start+1] * np.pi])
        start += 2

        for _ in range(self.num_helicopters):
            detected = observations[start]
            heli = Heli(detected, observations[start+1:start+3])
            self.heli_list.append(heli)
            start += 3

        for _ in range(self.num_search_parties):
            detected = observations[start]
            search_party = SearchParty(detected, observations[start+1:start+3])
            self.search_party_list.append(search_party)
            start += 3

        # wrapped_obs = self.obs_name(wrapped_obs)
        # for _ in range(self.num_rendezvous_points):
            # wrapped_obs[f"rendezvous_{i}"]
        self.rendezvous_point = RendezvousPoint(observations[start:start+2])
        start += 2

        # self.goal_location = np.rint(observations[start:start+2] * np.array([DIM_X, DIM_Y]))

        # remove cameras that are on top of a known hideout
        # for camera in self.camera_list:
        #     for hideout in self.known_hideout_list:
        #         if np.linalg.norm(camera.location - hideout.location) < 1:
        #             self.camera_list.remove(camera)

    def detected_helicopter(self):
        for heli in self.heli_list:
            if heli.detected:
                return heli
        return False

    def detected_search_party(self):
        for search_party in self.search_party_list:
            if search_party.detected:
                return search_party
        return False

    def in_range_of_camera(self, speed):
        for camera in self.camera_list:
            if distance(camera.location, self.location) <= camera.max_pod_distance(speed):
                return camera
        return False

    def in_range_of_mountain(self, location):
        for mountain_location in self.terrain.mountain_locations:
            mountain_location_corrected = (mountain_location[1], mountain_location[0])
            # need distance from edge of mountain to center, 120?
            dist = distance(location, mountain_location_corrected)
            if dist <= MOUNTAIN_OUTER_RANGE:
                return dist, mountain_location_corrected
        return None, None

class DetectionObject:
    DIM_X = 0
    DIM_Y = 0
    def __init__(self, location, detection_object_type_coefficient):
        self.location = np.rint(location * np.array([self.DIM_X, self.DIM_Y]))
        self.detection_object_type_coefficient = detection_object_type_coefficient
        self.buffer_range = 5

        # self.detection_terrain_coefficient = {
        #     TerrainType.MOUNTAIN: 1.0,
        #     TerrainType.WOODS: 1.0,
        #     TerrainType.DENSE_FOREST: 0.5
        # }

    def __repr__(self) -> str:
        return "(" + str(self.location[0]) + ", " + str(self.location[1]) + ")"

    def base_100_pod_distance(self, speed):
        """
        Calculate the distance within which the Probability of Detection is 100%
        :param speed: the speed of the detected object
        :return: the maximum distance of 100% PoD
        """
        # cameras can detect an object within 4 grids moving with speed 1 with 100% PoD in wood
        # return 4 * self.detection_terrain_coefficient[self.terrain.terrain_given_location(self.location)] * self.detection_object_type_coefficient * speed
        return 4 * self.detection_object_type_coefficient * speed

    def max_pod_distance(self, speed):
        return self.base_100_pod_distance(speed) * 3 + self.buffer_range

class Camera(DetectionObject):
    def __init__(self, location):
        super().__init__(location, 1)

class Heli(DetectionObject):
    def __init__(self, detected, location):
        self.detected = detected
        super().__init__(location, 0.5)

class SearchParty(DetectionObject):
    def __init__(self, detected, location):
        self.detected = detected
        super().__init__(location, 0.75)


class Hideout:
    DIM_X = 0
    DIM_Y = 0
    def __init__(self, type_hideout, location):
        self.type_hideout = type_hideout
        self.location = np.rint(location * np.array([self.DIM_X, self.DIM_Y]))

    def __repr__(self) -> str:
        return "(" + str(self.location[0]) + ", " + str(self.location[1]) + ")"

class RendezvousPoint:
    DIM_X = 0
    DIM_Y = 0
    def __init__(self, location):
        self.location = np.rint(location * np.array([self.DIM_X, self.DIM_Y]))

    def __repr__(self) -> str:
        return "(" + str(self.location[0]) + ", " + str(self.location[1]) + ")"