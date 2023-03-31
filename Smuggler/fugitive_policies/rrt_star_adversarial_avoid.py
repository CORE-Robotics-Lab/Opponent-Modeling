import numpy as np

from .utils import clip_theta, distance, c_str
import matplotlib.pyplot as plt
import time
from fugitive_policies.custom_queue import QueueFIFO
from fugitive_policies.rrt_star_adversarial_heuristic import RRTStarAdversarial, Plotter

DIM_X = 2428
DIM_Y = 2428

# MOUNTAIN_OUTER_RANGE = 150
MOUNTAIN_INNER_RANGE = 155
import math
import random

class RRTStarAdversarialAvoid(RRTStarAdversarial):
    def __init__(self, env,             
            n_iter=1500, 
            step_len=150, 
            search_radius=150, 
            max_speed=15,
            terrain_cost_coef=500, 
            visualize=False,
            gamma=15, 
            goal_sample_rate=0.1,
            epsilon = 0.1,):

        super().__init__(env, n_iter, step_len, search_radius, max_speed, terrain_cost_coef, visualize, gamma, goal_sample_rate)
        self.epsilon = epsilon
        self.DEBUG = False
        self.MIN_DIST_TO_HIDEOUT = 50
        self.mountain_travel = "optimal"
        self.reset()

    def reset(self):
        self.reset_plan()
        self.current_behavior = None
        self.current_hideout_goal = None
        self.behaviors = ['evade heli', 'evade search party', 'speed to known hideout',
                          'speed to unknown hideout']
        self.last_action = None
        self.behavior_completed = False
        self.being_tracked_for_n_timesteps = []

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

    def straight_line_action_to_closest_unknown_hideout(self):
        _, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
        theta = self.calculate_desired_heading(self.observations.location, closest_unknown_hideout.location)
        dist = distance(self.observations.location, closest_unknown_hideout.location)
        speed = np.clip(dist, 0, 7.5)
        self.current_hideout_goal = closest_unknown_hideout
        return np.array([speed, theta], dtype=np.float32)

    def action_to_closest_unknown_hideout(self, plot=True):
        """ Uses RRT* to plan path to closest unknown hideout """

        mountain_dist, mountain_in_range = self.observations.in_range_of_mountain(self.observations.location)
        if mountain_in_range:
            # if we're within a mountain, don't use rrt*
            return self.wrapper_avoid_mountain(self.straight_line_action_to_closest_unknown_hideout())

        # self.observations.process_observation(observation)
        if len(self.actions) == 0:
            closest_known_hideout, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
            self.current_hideout_goal = closest_unknown_hideout
            goal_location = closest_unknown_hideout.location
            start = time.time()
            path = self.plan(goal_location)
            print("Planning time:", time.time() - start)
            startpos = (self.s_start.x, self.s_start.y)
            endpos = (self.s_goal.x, self.s_goal.y)
            if plot:
                plotter = Plotter(self.terrain, startpos, endpos, plot_mountains=False, live=False)
                plotter.create_graph(self.s_start, background_img=self.env.cached_terrain_image)
                # plotter.create_path_node(self.goal)
                plotter.create_path(path)
                plt.savefig(f'figures/path_rrt_star_{self.terrain_cost_coef}.png', dpi=600)
                plt.close()
                # plt.show()
            self.actions = self.convert_path_to_actions(path)
        return self.actions.pop(0)

    def reset_plan(self):
        """ Removes our RRT* plan """
        self.actions = []

    def get_desired_action(self, observation):
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

        # if detected_helicopter or detected_search_party:
            # print("Detected helicopter or search party")

        if camera_in_range and detected_helicopter and detected_search_party:
            # zoom to the closest known hideout
            theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
            desired_action = self.wrapper_avoid_mountain(np.array([7.5, theta], dtype=np.float32))
        elif detected_helicopter and detected_search_party and np.sum(self.being_tracked_for_n_timesteps[-100:]) > 50:
            if self.DEBUG:
                print('you have been tracked for too long and failed')
            # zoom to the closest known hideout
            theta = self.calculate_desired_heading(self.observations.location, closest_known_hideout.location)
            self.reset_plan()
            desired_action = self.wrapper_avoid_mountain(np.array([7.5, theta], dtype=np.float32))

        # only detected by helicopter and have seen it in 7/10 of the last 10 timesteps
        elif detected_helicopter and np.sum(self.being_tracked_for_n_timesteps[-10:]) >= 7:
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
                        self.reset_plan()
                        desired_action = self.wrapper_avoid_mountain(np.array([7.5, self.current_behavior_heading], dtype=np.float32))
                        if np.sum(self.being_tracked_for_n_timesteps[-200:]) > 180:
                            if self.DEBUG:
                                print('you have been tracked for too long and failed')
                            # zoom to the closest known hideout
                            theta = self.calculate_desired_heading(self.observations.location,
                                                                   closest_known_hideout.location)
                            desired_action = self.wrapper_avoid_mountain(np.array([7.5, theta], dtype=np.float32))
                    else:
                        if self.DEBUG:
                            print('you have almost evaded, slow down, change direction')
                        self.reset_plan()
                        desired_action = self.wrapper_avoid_mountain(self.action_to_different_unknown_hideout(self.current_hideout_goal))
                        self.current_behavior_heading = desired_action[1]
                        self.last_action = 'you have almost evaded, slow down, change direction'
                elif self.current_behavior == self.behaviors[0]:
                    # you are evading, but not in dense forest yet
                    if self.DEBUG:
                        print('you are evading, but not in dense forest yet')
                    self.reset_plan()
                    desired_action = self.wrapper_avoid_mountain(np.array([7.5, self.current_behavior_heading], dtype=np.float32))
                    if np.sum(self.being_tracked_for_n_timesteps[-200:]) > 180:
                        if self.DEBUG:
                            print('you have been tracked for too long and failed')
                        # zoom to the closest known hideout
                        theta = self.calculate_desired_heading(self.observations.location,
                                                               closest_known_hideout.location)
                        self.reset_plan()
                        desired_action = self.wrapper_avoid_mountain(np.array([7.5, theta], dtype=np.float32))
                else:
                    # start evading, determine direction to go
                    if self.DEBUG:
                        print('start evading, determine direction to go')
                    theta = self.calculate_evasive_angle(self.observations.location, self.env.hideout_list)
                    self.current_behavior = self.behaviors[0]
                    self.current_behavior_heading = theta
                    self.reset_plan()
                    desired_action = self.wrapper_avoid_mountain(np.array([5, theta], dtype=np.float32))

        elif detected_search_party and np.sum(self.being_tracked_for_n_timesteps[-10:]) >= 7:
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
                    self.reset_plan()
                    # you have almost evaded, slow down, change direction
                    if self.last_action == 'you have almost evaded, slow down, change direction':
                        desired_action = self.wrapper_avoid_mountain(np.array([2, self.current_behavior_heading], dtype=np.float32))
                    else:
                        if self.DEBUG:
                            print('you have almost evaded, slow down, change direction')
                        self.reset_plan()
                        desired_action = self.wrapper_avoid_mountain(self.action_to_different_unknown_hideout(self.current_hideout_goal))
                        self.last_action = 'you have almost evaded, slow down, change direction'
                elif self.current_behavior == self.behaviors[1]:
                    # you are evading, but not in dense forest yet
                    if self.DEBUG:
                        print('you are evading, but not in dense forest yet')
                    self.reset_plan()
                    desired_action = self.wrapper_avoid_mountain(np.array([7.5, self.current_behavior_heading], dtype=np.float32))
                    
                else:
                    # start evading, determine direction to go
                    if self.DEBUG:
                        print('start evading, determine direction to go')
                    theta = self.calculate_evasive_angle(self.observations.location, self.env.hideout_list)
                    self.current_behavior = self.behaviors[1]
                    self.current_behavior_heading = theta
                    self.reset_plan()
                    desired_action = np.array([7.5, theta], dtype=np.float32)
        else:
            if self.DEBUG:
                print('you are detecting nothing, move to closest hideout with rrt star')
            desired_action = self.action_to_closest_unknown_hideout()
        # import time
        # time.sleep(1)
        if self.DEBUG:
            print('---------------------------')
        return desired_action

    def wrapper_avoid_mountain(self, desired_action):
        """Takes a desired action and ensures we don't hit mountain with it """
        new_location = self.simulate_action(self.observations.location, desired_action)
        mountain_dist, mountain_in_range = self.observations.in_range_of_mountain(new_location)

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
    
    def calculate_desired_heading(self, start_location, end_location):
        return np.arctan2(end_location[1] - start_location[1], end_location[0] - start_location[0])

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

    def update_tracking(self, detected_helicopter, detected_search_party, camera_in_range):
        if detected_helicopter or detected_search_party or camera_in_range:
            self.being_tracked_for_n_timesteps.append(1)
        else:
            self.being_tracked_for_n_timesteps.append(0)

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

    def predict(self, observation, deterministic=True):
        return (self.get_desired_action(observation), None)