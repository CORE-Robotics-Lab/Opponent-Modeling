import numpy as np
from random import random
from .rrt import Graph, dijkstra

from simulator import search_party
from .utils import clip_theta, distance, pick_closer_theta

DIM_X = 2428
DIM_Y = 2428

MOUNTAIN_OUTER_RANGE = 150
MOUNTAIN_INNER_RANGE = 140

class DetectionObject:
    def __init__(self, location, detection_object_type_coefficient):
        self.location = np.rint(location * np.array([DIM_X, DIM_Y]))
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
    def __init__(self, type_hideout, location):
        self.type_hideout = type_hideout
        self.location = np.rint(location * np.array([DIM_X, DIM_Y]))

    def __repr__(self) -> str:
        return "(" + str(self.location[0]) + ", " + str(self.location[1]) + ")"


class HeuristicPolicyGoal:
    def __init__(self, env):
        self.env = env
        self.terrain = self.env.terrain
        self.dim_x = env.terrain.dim_x
        self.dim_y = env.terrain.dim_y
        self.num_known_cameras = env.num_known_cameras
        self.num_search_parties = env.num_search_parties
        self.num_helicopters = env.num_helicopters
        self.num_known_hideouts = env.num_known_hideouts
        self.num_unknown_hideouts = env.num_unknown_hideouts
        self.max_timesteps = 4320  # 72 hours = 4320 minutes
        self.observations = self.Observation(self.terrain, self.num_known_cameras, self.num_helicopters, self.num_known_hideouts, self.num_unknown_hideouts, self.num_search_parties)
        self.first_run = True

    class Observation:
        """ Observation data 
        0: timestep
        n known cameras (location[0], location[1]) scaled by size of board
        fugitive location (x, y) scaled by size of board
        fugitive velocity (x, y) scaled by 15 (max speed) and np.pi for heading
        fugitive detection of parties
        local terrain (need full terrain so pass in env.terrain)
        """
        def __init__(self, terrain, num_known_cameras, num_helicopters, num_known_hideouts, num_unknown_hideouts, num_search_parties):
            self.terrain = terrain
            self.num_known_cameras = num_known_cameras
            self.num_helicopters = num_helicopters
            self.num_known_hideouts = num_known_hideouts
            self.num_unknown_hideouts = num_unknown_hideouts
            self.num_search_parties = num_search_parties
            
            self.location = np.zeros(2)
            self.goal_location = np.zeros(2) 
            self.camera_list = []
            self.heli_list = []
            self.unknown_hideout_list = []
            self.known_hideout_list = []
            self.search_party_list = []

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

            self.location = np.rint(observations[start:start+2] * np.array([DIM_X, DIM_Y]))
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

            self.goal_location = np.rint(observations[start:start+2] * np.array([DIM_X, DIM_Y]))

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

    def get_angles_away_from_object_location(self, object_location, start_location):
        theta = self.calculate_desired_heading(start_location, object_location)
        return clip_theta(theta - np.pi/2), clip_theta(theta + np.pi/2)
    

    def get_closest_hideouts(self, location):
        def get_closest_hideout(location, hideout_list):
            min_dist = np.inf
            closest_hideout = None
            for hideout in hideout_list:
                dist = distance(location, hideout.location)
                if dist < min_dist:
                    min_dist = dist
                    closest_hideout = hideout
            return closest_hideout

        closest_known_hideout = get_closest_hideout(location, self.observations.known_hideout_list)
        closest_unknown_hideout = get_closest_hideout(location, self.observations.unknown_hideout_list)

        return closest_known_hideout, closest_unknown_hideout

    def simulate_action(self, start_location, action):
        direction = np.array([np.cos(action[1]), np.sin(action[1])])
        speed = action[0]
        new_location = np.round(start_location + direction * speed)
        new_location[0] = np.clip(new_location[0], 0, self.dim_x - 1)
        new_location[1] = np.clip(new_location[1], 0, self.dim_y - 1)
        new_location = new_location.astype(np.int)
        return new_location

    def newVertex(self, randvex, nearvex):
        dirn = np.array(randvex) - np.array(nearvex)
        length = np.linalg.norm(dirn)
        dirn = (dirn / length) * min(15, length)
        speed = min(14.5, length) # rounding sometimes makes this go over, use 14.5 to be safe 
        theta = np.arctan2(dirn[1], dirn[0])
        action = np.array([speed, theta])
        newvex = self.simulate_action(nearvex, action)
        return newvex


    def calculate_desired_heading(self, start_location, end_location):
        return np.arctan2(end_location[1] - start_location[1], end_location[0] - start_location[0])

    def check_collision(self, location):
        return self.terrain.world_representation[0, location[0], location[1]] == 1

    def nearest(self, G, vex):
        Nvex = None
        Nidx = None
        minDist = float("inf")

        for idx, v in enumerate(G.vertices):
            # # line = Line(v, vex)

            # if self.check_collision():
            #     continue

            dist = distance(np.asarray(v), np.asarray(vex))
            if dist < minDist:
                minDist = dist
                Nidx = idx
                Nvex = v

        return Nvex, Nidx

    def plan(self, n_iter=1000, stepSize=15):
        ''' RRT algorithm '''
        startpos = tuple(map(int, self.observations.location))
        endpos = tuple(map(int, self.observations.goal_location))
        radius = 10
        G = Graph(startpos, endpos)

        for _ in range(n_iter):
            if random() < 0.65:
                randvex = G.randomPositionBiased()
            else:
                randvex = np.asarray(endpos)

            if self.check_collision(randvex):
                continue

            nearvex, nearidx = self.nearest(G, randvex)
            if nearvex is None or tuple(randvex) == nearvex:
                continue

            newvex = self.newVertex(randvex, nearvex)
            if self.check_collision(newvex):
                continue

            newidx = G.add_vex(tuple(newvex))
            dist = distance(newvex, nearvex)
            G.add_edge(newidx, nearidx, dist)

            dist = distance(newvex, G.endpos)
            # print(dist)
            if dist < 15:
                endidx = G.add_vex(G.endpos)
                G.add_edge(newidx, endidx, dist)
                G.success = True
                # print('success')
                # break
        if G.success:
            path = dijkstra(G)
            return path
        else:
            return None

    # def get_action(self,observation):
    #     desired_action = self.get_action_desired(observation)
    #     new_location = self.simulate_action(self.observations.location, desired_action)

    #     if self.terrain.world_representation[0, new_location[0], new_location[1]] == 1:
    #         mountain_in_range = (new_location[0], new_location[1])
    #         goal_headings = self.get_angles_away_from_object_location(mountain_in_range, self.observations.location)
    #         theta = self.calculate_desired_heading(self.observations.location, new_location)
    #         desired_heading = pick_closer_theta(theta, goal_headings)            
    #         return np.array([15, desired_heading], dtype=np.float32)

    #     return desired_action

    def arctan_clipped(self, loc1, loc2):
        heading = np.arctan2(loc2[1] - loc1[1], loc2[0] - loc1[0])
        if heading < -np.pi:
            heading += 2 * np.pi
        elif heading > np.pi:
            heading -= 2 * np.pi
        return heading

    def get_angle_away(self, mountain_in_range, location, goal_location):
        if np.array_equal(self.observations.goal_location, np.array([140., 200.])): # Goal 0
            location_theta = np.arctan2(location[1] - goal_location[1], location[0] - goal_location[0])
            mountain_theta = np.arctan2(mountain_in_range[1] - goal_location[1], mountain_in_range[0] - goal_location[0])
            location_to_mountain_theta = np.arctan2(location[1] - mountain_in_range[1], location[0] - mountain_in_range[0])
        else: # Goal 1/2
            location_theta = self.arctan_clipped(location, goal_location)
            mountain_theta = self.arctan_clipped(mountain_in_range, goal_location)
            location_to_mountain_theta = self.arctan_clipped(location, mountain_in_range)

        theta = self.calculate_desired_heading(location, mountain_in_range)

        # mod_nearest_int = location_to_mountain_theta - location_theta * round(location_to_mountain_theta / location_theta)
        if abs(location_to_mountain_theta - location_theta) > np.pi/2:
            return self.calculate_desired_heading(location, goal_location)

        if location_theta < mountain_theta:
            return clip_theta(theta + np.pi/2)
        else:
            return clip_theta(theta - np.pi/2)

    def get_action(self, observation):
        self.observations.process_observation(observation)
        desired_hideout_location = self.observations.goal_location

        desired_action = self.action_to_desired_location(desired_hideout_location)
        new_location = self.simulate_action(self.observations.location, desired_action)

        mountain_dist, mountain_in_range = self.observations.in_range_of_mountain(new_location)

        if mountain_in_range:
            if mountain_dist <= MOUNTAIN_INNER_RANGE:
                # move perpendicular to mountain range
                theta = self.calculate_desired_heading(self.observations.location, mountain_in_range)
                if theta < 0:
                    theta += np.pi
                else:
                    theta -= np.pi
                desired_action = np.array([7.5, theta], dtype=np.float32)
            else:
                desired_heading = self.get_angle_away(mountain_in_range, self.observations.location, desired_hideout_location)
                desired_action = np.array([7.5, desired_heading], dtype=np.float32)
        
        return desired_action

    def predict(self, observation, deterministic=True):
        return (self.get_action(observation), None)

    def action_to_desired_location(self, location):
        theta = self.calculate_desired_heading(self.observations.location, location)
        dist = distance(self.observations.location, location)
        speed = np.clip(dist, 0, 7.5)
        return np.array([speed, theta], dtype=np.float32)  

    def action_to_closest_known_hideout(self):
        closest_known_hideout, _ = self.get_closest_hideouts(self.observations.location)
        theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
        dist = distance(self.observations.location, closest_known_hideout.location)
        speed = np.clip(dist, 0, 7.5)
        return np.array([speed, theta], dtype=np.float32)

    def action_to_closest_unknown_hideout(self):
        _, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
        theta = self.calculate_desired_heading(self.observations.location, closest_unknown_hideout.location)
        dist = distance(self.observations.location, closest_unknown_hideout.location)
        speed = np.clip(dist, 0, 7.5)
        return np.array([speed, theta], dtype=np.float32)
