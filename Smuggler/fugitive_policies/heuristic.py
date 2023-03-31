import numpy as np
import random

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


class HeuristicPolicy:
    def __init__(self, env, random_mountain=False, mountain_travel="optimal", epsilon=0):
        """ 
        :param env: the environment to use for the policy
        :param random_mountain: whether to travel optimally around the mountain
        :param mountain_direction: the direction to travel around the mountain (optimal, left, right)
        :param epsilon: Using epsilon-greedy policy, epsilon is how often to take a random action

        """
        self.env = env
        self.terrain = self.env.unwrapped.terrain
        self.dim_x = env.terrain.dim_x
        self.dim_y = env.terrain.dim_y
        self.num_known_cameras = env.num_known_cameras
        self.num_search_parties = env.num_search_parties
        self.num_helicopters = env.num_helicopters
        self.num_known_hideouts = env.num_known_hideouts
        self.num_unknown_hideouts = env.num_unknown_hideouts
        self.max_timesteps = 4320  # 72 hours = 4320 minutes
        self.MAX_TIME = 1000
        self.MIN_DIST_TO_HIDEOUT = 50
        self.observations = self.Observation(self.terrain, self.num_known_cameras, self.num_helicopters,
                                             self.num_known_hideouts, self.num_unknown_hideouts,
                                             self.num_search_parties)
        # adding tracking for multi-step behaviors
        self.current_behavior = None
        self.current_hideout_goal = None
        self.behaviors = ['evade heli', 'evade search party', 'speed to known hideout',
                          'speed to unknown hideout']
        self.last_action = None
        self.behavior_completed = False
        self.being_tracked_for_n_timesteps = []
        self.epsilon = epsilon
        self.DEBUG = False

        # If optimal_mountain, we travel shortest path around it
        # If suboptimal_mountain, we travel longest path around it
        if random_mountain:
            self.mountain_travel = random.choice(["optimal", "left", "right"])
        else:
            self.mountain_travel = mountain_travel

    class Observation:
        """ Observation data
        0: timestep
        n known cameras (location[0], location[1]) scaled by size of board
        fugitive location (x, y) scaled by size of board
        fugitive velocity (x, y) scaled by 15 (max speed) and np.pi for heading
        fugitive detection of parties
        local terrain (need full terrain so pass in env.terrain)
        """

        def __init__(self, terrain, num_known_cameras, num_helicopters, num_known_hideouts, num_unknown_hideouts,
                     num_search_parties):
            self.terrain = terrain
            self.num_known_cameras = num_known_cameras
            self.num_helicopters = num_helicopters
            self.num_known_hideouts = num_known_hideouts
            self.num_unknown_hideouts = num_unknown_hideouts
            self.num_search_parties = num_search_parties

            self.location = np.zeros(2)
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
            self.timestep = observations[0] * 4320
            start = 1
            for _ in range(self.num_known_cameras):
                camera = Camera(observations[start:start + 2])
                self.camera_list.append(camera)
                start += 2

            for _ in range(self.num_known_hideouts + self.num_unknown_hideouts):
                hideout = Hideout(observations[start], observations[start + 1:start + 3])
                if observations[start] == 1:
                    self.known_hideout_list.append(hideout)
                else:
                    self.unknown_hideout_list.append(hideout)
                start += 3

            self.location = np.rint(observations[start:start + 2] * np.array([DIM_X, DIM_Y]))
            start += 2

            self.action = np.array([observations[start] * 15, observations[start + 1] * np.pi])
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
            # for mountain_location in self.terrain.mountain_locations:
            #     mountain_location_corrected = (mountain_location[1], mountain_location[0])
            #     # need distance from edge of mountain to center, 120?
            #     dist = distance(location, mountain_location_corrected)
            #     if dist <= MOUNTAIN_OUTER_RANGE:
            #         return dist, mountain_location_corrected
            return None, None

    def get_angles_away_from_object_location(self, object_location, start_location):
        theta = self.calculate_desired_heading(start_location, object_location)
        return clip_theta(theta - np.pi / 2), clip_theta(theta + np.pi / 2)

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

    def calculate_desired_heading(self, start_location, end_location):
        return np.arctan2(end_location[1] - start_location[1], end_location[0] - start_location[0])

    def check_collision(self, location):
        return self.terrain.world_representation[0, location[0], location[1]] == 1

    def plan(self, start, goal):
        print(start, goal)

    def arctan_clipped(self, loc1, loc2):
        heading = np.arctan2(loc2[1] - loc1[1], loc2[0] - loc1[0])
        if heading < -np.pi:
            heading += 2 * np.pi
        elif heading > np.pi:
            heading -= 2 * np.pi
        return heading

    def get_angle_away(self, mountain_in_range, location, theta):
        # location_to_mountain_theta = self.arctan_clipped(location, mountain_in_range)
        # location_to_mountain_theta = np.arctan2(location[1] - mountain_in_range[1], location[0] - mountain_in_range[0])
        location_to_mountain_theta = np.arctan2(mountain_in_range[1] - location[1], mountain_in_range[0] - location[0])
        if -np.pi < location_to_mountain_theta < -np.pi / 2:
            theta_one = location_to_mountain_theta + np.pi / 2
            theta_two = location_to_mountain_theta + 3 * np.pi / 2
            # in bottom left quadrant, have to adjust bounds
            if theta < theta_one or theta > theta_two:
                # need to move away from mountain
                # print("move away 3")
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2 * np.pi - theta_one),
                                     np.abs(theta - 2 * np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2 * np.pi - theta_two),
                                     np.abs(theta - 2 * np.pi - theta_two))
                
                if self.mountain_travel == "optimal":
                    if theta_dist_one < theta_dist_two:
                        return theta_one
                    else:
                        return theta_two
                elif self.mountain_travel == "left":
                    return theta_two
                else:
                    return theta_one
                # return clip_theta(location_to_mountain_theta - np.pi/2)
            else:
                # print("move towards 3")
                return theta
        elif np.pi / 2 < location_to_mountain_theta < np.pi:
            theta_one = location_to_mountain_theta - np.pi / 2
            theta_two = location_to_mountain_theta - 3 * np.pi / 2
            # in bottom right quadrant
            if theta > theta_one or theta < theta_two:
                # need to move away from mountain
                # print("move away 2")
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2 * np.pi - theta_one),
                                     np.abs(theta - 2 * np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2 * np.pi - theta_two),
                                     np.abs(theta - 2 * np.pi - theta_two))
                if self.mountain_travel == "optimal":
                    if theta_dist_one < theta_dist_two:
                        return theta_one
                    else:
                        return theta_two
                elif self.mountain_travel == "left":
                    return theta_one
                else:
                    return theta_two
            else:
                # print("move towards 2")
                return theta
        else:
            theta_one = location_to_mountain_theta - np.pi / 2
            theta_two = location_to_mountain_theta + np.pi / 2
            if theta_one < theta < theta_two:
                # print("move away 14")
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2 * np.pi - theta_one),
                                     np.abs(theta - 2 * np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2 * np.pi - theta_two),
                                     np.abs(theta - 2 * np.pi - theta_two))
                if self.mountain_travel == "optimal":
                    if theta_dist_one < theta_dist_two:
                        return theta_one
                    else:
                        return theta_two
                elif self.mountain_travel == "left":
                    return theta_one
                else:
                    return theta_two
            else:
                # print("move towards 14")
                return theta

    def calculate_evasive_angle(self, current_location, hideouts):
        """
        This function will look at the fugitives current location, hideouts, and terrain, and choose a direction to go
        to evade detection (into the forest)
        :param current_location:
        :param hideouts:
        :return:
        """
        # find locations where forest is pretty dense
        dense_forest = self.terrain.forest_density_array < .4

        # check around some fixed region of the fugitive
        ran = 250
        lb_x = max(int(current_location[0] - ran), 0) # lower bound x
        ub_x = min(int(current_location[0] + ran), DIM_X) # upper bound x
        lb_y = max(int(current_location[1] - ran), 0) # lower bound y
        ub_y = min(int(current_location[1] + ran), DIM_Y) # upper bound y
        best_dist = np.inf
        candidate = None
        for i in range(lb_x, ub_x):
            for j in range(lb_y, ub_y):
                if i == current_location[0] and j== current_location[1]:
                    continue
                s = (i, j)
                dist = np.linalg.norm(s - current_location)
                # if its a patch of forest and not a one off
                if np.sum(dense_forest[i - 3:i + 3, j - 3:j + 3]) < 18:
                    continue
                if dist <= best_dist:
                    best_dist = dist
                    candidate = s

        if candidate is None:
            candidate = (1500, 1500)
        
        angle = self.calculate_desired_heading(current_location, candidate)
        if self.DEBUG:
            print("Candidate location is: ", candidate)

        return angle

    def in_dense_forest(self,  current_location):
        dense_forest = self.terrain.forest_density_array < .4
        i = int(current_location[0])
        j = int(current_location[1])

        if np.sum(dense_forest[i - 3:i + 3, j - 3:j + 3]) > 17:
            if self.DEBUG:
                print(current_location, ' in dense forest.')
            return True
        else:
            if self.DEBUG:
                print(current_location, ' NOT in dense forest.')
            return False

    def get_desired_action(self, observation):
        # Needs some tuning because fugitive cannot detect anything when running more than speed 8
        # Therefore sometimes the prisoner will stutter by sprinting, then detecting nothing, then sprinting

        self.observations.process_observation(observation)

        detected_helicopter = self.observations.detected_helicopter()
        detected_search_party = self.observations.detected_search_party()

        closest_known_hideout, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
        theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
        camera_in_range = self.observations.in_range_of_camera(self.observations.action[0])
        if self.DEBUG:
            print('t=', self.env.timesteps)
            print('fugitive current location', self.observations.location)
            print('detected helicopter', detected_helicopter)
            print('detected search party ',  detected_search_party)
            print('camera in range', camera_in_range)
        self.update_tracking(detected_helicopter, detected_search_party, camera_in_range)
        # if self.env.timesteps == 23:
        #     print('stop')
        # if self.observations.location[0] == 2427:
        #     print('here')

        # if detected_helicopter or detected_search_party:
            # print("Detected helicopter or search party")

        if camera_in_range and detected_helicopter and detected_search_party:
            # zoom to the closest known hideout
            theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
            desired_action = np.array([7.5, theta], dtype=np.float32)
        elif detected_helicopter and detected_search_party and np.sum(self.being_tracked_for_n_timesteps[-100:]) > 50:
            if self.DEBUG:
                print('you have been tracked for too long and failed')
            # zoom to the closest known hideout
            theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
            desired_action = np.array([7.5, theta], dtype=np.float32)

        # only detected by helicopter
        elif detected_helicopter:
            # check if you are in dense forest
            in_dense_forest = self.in_dense_forest(self.observations.location)
            if distance(self.observations.location, detected_helicopter.location) > 100:
                if self.DEBUG:
                    print('dont worry about heli, its too far')
                self.last_action = 'dont worry about heli, its too far'
                desired_action = self.action_to_closest_unknown_hideout()

            elif in_dense_forest and distance(self.observations.location, detected_helicopter.location) > 50:
                # you have distanced yourself enough away from the heli
                if self.DEBUG:
                    print('you have distanced yourself enough away from the heli')
                desired_action = self.action_to_closest_unknown_hideout()
            else:
                # start evading, or continue evading
                if in_dense_forest and self.current_behavior == self.behaviors[0]:
                    # you have almost evaded, slow down, change direction
                    if self.last_action == 'you have almost evaded, slow down, change direction':
                        desired_action = np.array([7.5, self.current_behavior_heading], dtype=np.float32)
                        if np.sum(self.being_tracked_for_n_timesteps[-200:]) > 180:
                            if self.DEBUG:
                                print('you have been tracked for too long and failed')
                            # zoom to the closest known hideout
                            theta = self.calculate_desired_heading(self.observations.location,
                                                                   closest_known_hideout.location)
                            desired_action = np.array([7.5, theta], dtype=np.float32)
                    else:
                        if self.DEBUG:
                            print('you have almost evaded, slow down, change direction')
                        desired_action = self.action_to_different_unknown_hideout(self.current_hideout_goal)
                        self.current_behavior_heading = desired_action[1]
                        self.last_action = 'you have almost evaded, slow down, change direction'
                elif self.current_behavior == self.behaviors[0]:
                    # you are evading, but not in dense forest yet
                    if self.DEBUG:
                        print('you are evading, but not in dense forest yet')
                    desired_action = np.array([7.5, self.current_behavior_heading], dtype=np.float32)
                    if np.sum(self.being_tracked_for_n_timesteps[-200:]) > 180:
                        if self.DEBUG:
                            print('you have been tracked for too long and failed')
                        # zoom to the closest known hideout
                        theta = self.calculate_desired_heading(self.observations.location,
                                                               closest_known_hideout.location)
                        desired_action = np.array([7.5, theta], dtype=np.float32)
                else:
                    # start evading, determine direction to go
                    if self.DEBUG:
                        print('start evading, determine direction to go')
                    theta = self.calculate_evasive_angle(self.observations.location, self.env.hideout_list)
                    self.current_behavior = self.behaviors[0]
                    self.current_behavior_heading = theta
                    desired_action = np.array([5, theta], dtype=np.float32)

        elif detected_search_party:
            # speed up and try and run away from search parties
            if self.DEBUG:
                print('speed up and try and run away from search parties')
            in_dense_forest = self.in_dense_forest(self.observations.location)
            if distance(self.observations.location, detected_search_party.location) > 50:
                if self.DEBUG:
                    print('dont worry about search party, its too far')
                self.last_action = 'dont worry about heli, its too far'
                desired_action = self.action_to_closest_unknown_hideout()
            elif in_dense_forest and distance(self.observations.location, detected_search_party.location) > 20:
                # you have distanced yourself enough away from the heli
                if self.DEBUG:
                    print('you have distanced yourself enough away from the search part')
                desired_action = self.action_to_closest_unknown_hideout()
            else:
                # start evading, or continue evading
                if in_dense_forest and self.current_behavior == self.behaviors[0]:
                    # you have almost evaded, slow down, change direction
                    if self.last_action == 'you have almost evaded, slow down, change direction':
                        desired_action = np.array([2, self.current_behavior_heading], dtype=np.float32)
                    else:
                        if self.DEBUG:
                            print('you have almost evaded, slow down, change direction')
                        desired_action = self.action_to_different_unknown_hideout(self.current_hideout_goal)
                        self.last_action = 'you have almost evaded, slow down, change direction'
                elif self.current_behavior == self.behaviors[1]:
                    # you are evading, but not in dense forest yet
                    if self.DEBUG:
                        print('you are evading, but not in dense forest yet')
                    desired_action = np.array([7.5, self.current_behavior_heading], dtype=np.float32)
                else:
                    # start evading, determine direction to go
                    if self.DEBUG:
                        print('start evading, determine direction to go')
                    theta = self.calculate_evasive_angle(self.observations.location, self.env.hideout_list)
                    self.current_behavior = self.behaviors[1]
                    self.current_behavior_heading = theta
                    desired_action = np.array([7.5, theta], dtype=np.float32)
            # theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
            # desired_action = np.array([12, theta], dtype=np.float32)
        # elif camera_in_range:
        #     theta = self.calculate_desired_heading(self.observations.location, closest_unknown_hideout.location)
        #     # desired_heading = pick_closer_theta(theta, goal_headings)
        #     desired_heading = self.get_angle_away(camera_in_range.location, self.observations.location, theta)
        #     desired_action = np.array([7.5, desired_heading], dtype=np.float32)
        else:
            if self.DEBUG:
                print('you are detecting nothing, move to closest hideout')
            desired_action = self.action_to_closest_unknown_hideout()

        import time
        # time.sleep(1)
        if self.DEBUG:
            print('---------------------------')
        return desired_action

    def random_action(self):
        """ Return a random action"""
        return np.array([np.random.uniform(5, 15), np.random.uniform(-np.pi, np.pi)], dtype=np.float32)

    def get_action(self, observation):
        """ This is the main method called to move the Fugitive. We first calculate the desired action, then if a mountain is
        in the way, we choose an angle away from the mountain. """
        desired_action = self.get_desired_action(observation)
        new_location = self.simulate_action(self.observations.location, desired_action)
        mountain_dist, mountain_in_range = self.observations.in_range_of_mountain(new_location)

        if random.random() >= 1 - self.epsilon:
            desired_action = self.random_action()
            # new_speed = desired_action[0] + np.random.uniform(-5, 5)
            # new_direction = desired_action[1] + np.random.uniform(-np.pi, np.pi)
            # if new_direction > np.pi:
            #     new_direction -= 2 * np.pi
            # elif new_direction < -np.pi:
            #     new_direction += 2 * np.pi
            # desired_action = np.array([new_speed, new_direction], dtype=np.float32)

        # planning around mountains
        if mountain_in_range:
            # if we are within the inner bounds of the mountain, move directly outwards
            if mountain_dist <= MOUNTAIN_INNER_RANGE:
                # print("inner")
                theta = self.calculate_desired_heading(self.observations.location, mountain_in_range)
                if theta < 0:
                    theta += np.pi
                else:
                    theta -= np.pi
                desired_action = np.array([7.5, theta], dtype=np.float32)
            else:
                heading = desired_action[1]
                desired_heading = self.get_angle_away(mountain_in_range, self.observations.location, heading)
                desired_action = np.array([7.5, desired_heading], dtype=np.float32)

        _, distance_from_closest_hideout = self.get_closest_hideout(self.observations.location, self.observations.known_hideout_list + self.observations.unknown_hideout_list)
        if distance_from_closest_hideout < self.MIN_DIST_TO_HIDEOUT:
            desired_action = [0.0, 0.0]

        return desired_action

    def predict(self, observation, deterministic=True):
        return (self.get_action(observation), None)

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
        self.current_hideout_goal = closest_unknown_hideout
        return np.array([speed, theta], dtype=np.float32)

    def action_to_different_unknown_hideout(self, current_goal):
        hideout_distances = {}
        for hideout in self.observations.unknown_hideout_list:
            if (hideout.location == current_goal.location).all():
                continue
            dist = distance(self.observations.location, hideout.location)
            hideout_distances[hideout] = dist

        # choose closest distance hideout
        hid = min(hideout_distances.items(), key=lambda x:x[1])

        # hid = min(hideout_distances, key=hideout_distances.get())

        theta = self.calculate_desired_heading(self.observations.location, hid[0].location)
        self.current_hideout_goal = hid[0]
        self.current_behavior_heading = theta
        return np.array([1, theta], dtype=np.float32)

    def update_tracking(self, detected_helicopter, detected_search_party, camera_in_range):
        if detected_helicopter or detected_search_party or camera_in_range:
            self.being_tracked_for_n_timesteps.append(1)
        else:
            self.being_tracked_for_n_timesteps.append(0)
