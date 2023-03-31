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

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.cost = None
        # self.children = []
        self.children = set()

    def change_parent(self, new_parent_node):
        # if self in self.parent.children:
        self.parent.children.remove(self)
        self.parent = new_parent_node
        new_parent_node.children.add(self)

    @property
    def location(self):
        return np.array([self.x, self.y])

class RRTStarAdversarial:
    def __init__(self, env,             
            n_iter=1000, 
            step_len=150, 
            search_radius=150, 
            max_speed=7.5,
            terrain_cost_coef=500, 
            visualize=False,
            gamma=15, 
            goal_sample_rate=0.1):
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
        self.observations = Observation(self.terrain, self.num_known_cameras, self.num_helicopters, self.num_known_hideouts, self.num_unknown_hideouts, self.num_search_parties)
        self.first_run = True

        self.actions = []
        self.search_radius = search_radius
        self.iter_max = n_iter
        self.visualize = visualize
        self.vertex_pos = np.ones((5000, 2)) * np.inf # Initialize more than possible just in case
        self.terrain_cost_coef = terrain_cost_coef
        self.neighbor_gamma = gamma
        self.max_speed = max_speed
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate

    def get_angles_away_from_object_location(self, object_location, start_location):
        theta = self.calculate_desired_heading(start_location, object_location)
        return clip_theta(theta - np.pi/2), clip_theta(theta + np.pi/2)

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

    def check_collision_mountain(self, start_node, end_node):
        """ Check if the straight line path between the start and end location pass through the mountain.
            Rather than check every point in the mountain, we abstract as a circle and just check if 
            the points intersect the circle.

            Deprecated but useful to see math that is not vectorized in below function
        """
        a = np.array([start_node.x, start_node.y])
        b = np.array([end_node.x, end_node.y])
        
        for mountain_location in self.terrain.mountain_locations:
            c = np.array([mountain_location[1], mountain_location[0]])
            if distance(b, c) < MOUNTAIN_OUTER_RANGE:
                return True

            delta = a - b
            unit_vector = delta / np.linalg.norm(delta)
            d = np.cross(c - a, unit_vector)
            if -MOUNTAIN_OUTER_RANGE < d <  MOUNTAIN_OUTER_RANGE:
                return True

            if unit_vector @ a < unit_vector @ c < unit_vector @ b:
                return True
            if unit_vector @ b < unit_vector @ c < unit_vector @ a:
                return True
        return False

    def check_collision_mountain_vectorized(self, node, indices, vertices):
        """ 
            indices list is a numpy array of vertex indices corresponding to the vertices
            vertices is the numpy array of vertices where we check each between node and vertices to see if they are in the mountain
        """
        a = node.location
        # b = np.array([node_list[0].x, node_list[0].y])
        booleans = []
        for mountain_location in self.terrain.mountain_locations:
            c = np.array([mountain_location[1], mountain_location[0]])
            dists = np.linalg.norm(vertices - c, axis=1)
            dist_bools = dists < MOUNTAIN_OUTER_RANGE

            delta = a - vertices
            unit_vectors = delta / np.linalg.norm(delta, axis=1)[:, None]
            d = np.cross(c - a, unit_vectors)

            radius_bools = np.logical_and((-MOUNTAIN_OUTER_RANGE < d), (d < MOUNTAIN_OUTER_RANGE))
            # unit_vectors is of shape (num_vertices, 2)
            # vertices is of shape (num_vertices, 2)

            bool_one = np.logical_and((unit_vectors @ a < unit_vectors @ c), (unit_vectors @ c < (unit_vectors * vertices).sum(axis=1)))
            bool_two = np.logical_and(((unit_vectors*vertices).sum(axis=1) < unit_vectors @ c), (unit_vectors @ c < unit_vectors @ a))

            final_bool = np.logical_or.reduce((dist_bools, radius_bools, bool_one, bool_two))
            booleans.append(final_bool)

        booleans = np.logical_or.reduce(booleans)
        return indices[np.invert(booleans)]


    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((int(node_start.x + dist * math.cos(theta)),
                         int(node_start.y + dist * math.sin(theta))))

        return node_new, dist

    def repropagate_costs(self, node_parent, cost_diff):
        """
        For all children in node_parent's children list, update their costs by subtracting off cost_diff
        """
        OPEN = QueueFIFO()
        OPEN.put(node_parent)

        while not OPEN.empty():
            node = OPEN.get()

            if len(node.children) == 0:
                continue

            for node_c in node.children:
                # node_c.Cost = self.get_new_cost(node, node_c)
                # dist, _ = self.get_distance_and_angle(node, node_c)
                # node_c.cost = dist + self.terrain_cost(node_c) + node.cost
                new_cost = node_c.cost - cost_diff
                # node_c.cost = dist + self.terrain_cost_path(node, node_c) + node.cost
                node_c.cost = new_cost
                # print(new_cost)
                # assert new_cost == node_c.cost
                OPEN.put(node_c)

    def rewire(self, node_new, neighbor_index):
        """ Make the parent for a neighborhood node the new node that we just added? """
        for i in neighbor_index:
            node_neighbor = self.vertex[i]
            # if node_neighbor.parent == node_new:
            #     continue

            previous_cost = node_neighbor.cost
            dist, _ = self.get_distance_and_angle(node_new, node_neighbor)
            new_cost = dist + self.terrain_cost_path(node_new, node_neighbor) + node_new.cost # new cost for neighbor

            if previous_cost > new_cost:
                if self.visualize:
                    self.plotter.plot_edge(node_neighbor.parent, node_neighbor, color='white')

                node_neighbor.change_parent(node_new)
                node_neighbor.cost = new_cost
                cost_diff = previous_cost - new_cost
                self.repropagate_costs(node_neighbor, cost_diff)

                if self.visualize:
                    self.plotter.plot_edge(node_neighbor.parent, node_neighbor, color='red')


    def search_goal_parent(self):
        dist_list = self.get_dist_list(self.s_goal)
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.step_len]

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.vertex[i].cost + self.terrain_cost_path(self.vertex[i], self.s_goal) for i in node_index
                         if not self.check_collision_mountain(self.vertex[i], self.s_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def generate_random_node(self, goal_sample_rate):
        if np.random.random() > goal_sample_rate:
            delta = 50
            # return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
            #              np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

            return Node((int(np.random.uniform(0, self.dim_x)),
                         int(np.random.uniform(0, self.dim_y))))
            # xs = [self.s_start.x, self.s_goal.x]
            # ys = [self.s_start.y, self.s_goal.y]
            # return Node((int(np.random.uniform(min(xs), max(xs))),
            #              int(np.random.uniform(min(ys), max(ys)))))
        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.neighbor_gamma * self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)
        dist_table = self.get_dist_list(node_new)

        d = np.where(dist_table <= r)
        if len(d) == 1 and len(d[0]) == 0:
            # Just check if there's no results
            res = np.array([])
        else:
            indices = np.concatenate(d)
            vertices = self.vertex_pos[indices]
            res = self.check_collision_mountain_vectorized(node_new, indices, vertices)
        return res

    def terrain_cost(self, node):
        location = (node.x, node.y)
        return (self.env.terrain.detection_coefficient_given_location(location)) * self.terrain_cost_coef

    def terrain_cost_path(self, start_node, end_node):
        """ Rather computing the cost for both node, compute the cost for the whole path 
            include cost of start node but not the end node
        
        """
        vector = np.array([end_node.x - start_node.x, end_node.y - start_node.y])
        dist = np.linalg.norm(vector)
        norm = vector/dist
        num_points = dist // self.max_speed

        points = np.repeat(norm[np.newaxis, :], num_points, axis=0) # create an array of all normed points
        range_points = self.max_speed * np.arange(num_points)[:, np.newaxis] # create an array of single range
        terrain_points = np.einsum('ij,ik->ij', points, range_points) # 
        indices = np.round(terrain_points).astype('int')
        arr = self.terrain.world_representation[1, :, :]
        xs = indices[:, 0] + start_node.x
        ys = indices[:, 1] + start_node.y
        terrain_cost = self.terrain_cost_coef * np.sum(arr[xs, ys])
        return terrain_cost

    def get_dist_list(self, n):
        """ Get the distance between node n and every vertex in graph 
            returns a numpy array of distances
        """
        distances = np.linalg.norm(self.vertex_pos[:len(self.vertex)] - n.location, axis=1)
        return distances

    def nearest_neighbor(self, n):
        distances = self.get_dist_list(n)
        return self.vertex[int(np.argmin(distances))]

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def assign_parent(self, node_new, neighborhood):
        """ Assign parent to node given neighborhood indices"""
        cost_min = node_new.cost
        node_parent = node_new.parent
        # terrain_cost = self.terrain_cost(node_new)
        changed_parent = False

        for node_index in neighborhood:
            node_neighbor = self.vertex[node_index]
            dist, _ = self.get_distance_and_angle(node_new, node_neighbor)
            new_cost = dist + self.terrain_cost_path(node_new, node_neighbor) + node_neighbor.cost
            
            if new_cost < cost_min:
                changed_parent = True
                cost_min = new_cost
                node_parent = node_neighbor
        
        if changed_parent:
            node_new.change_parent(node_parent)
            # don't need to reproprogate because there should be no children node on node_new?
            node_new.cost = cost_min

    # @property 
    # def goal_not_found(self):
    #     pass        

    def plan(self, endpos):
        startpos = tuple(map(int, self.observations.location))
        endpos = tuple(map(int, endpos))
        self.s_start = Node(startpos)
        self.s_start.cost = 0
        self.s_goal = Node(endpos)
        self.vertex = [self.s_start]
        self.path = []
        self.vertex_pos[0] = np.array([startpos[0], startpos[1]])
        self.goal_not_found = True

        vertex_index = 1
        timer = 0

        if self.visualize:
            self.plotter = Plotter(self.terrain, startpos, endpos)
        k = 0
        # for k in range(self.iter_max):
        while k < self.iter_max or self.goal_not_found:
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(node_rand)
            node_new, dist = self.new_state(node_near, node_rand)

            if k % 100 == 0:
                # print(k, timer)
                timer = 0

            if self.check_collision_mountain(node_near, node_new):
                continue

            if node_new and not self.check_collision_mountain(node_near, node_new):
                node_new.parent = node_near
                node_near.children.add(node_new)
                node_new.cost = node_new.parent.cost + self.terrain_cost_path(node_new.parent, node_new) + dist

                neighbor_index = self.find_near_neighbor(node_new)

                self.vertex.append(node_new)
                self.vertex_pos[vertex_index] = np.array([node_new.x, node_new.y])
                if distance(node_new.location, self.s_goal.location) < self.step_len:
                    self.goal_not_found = False
                
                vertex_index += 1
                if len(neighbor_index) > 0:
                    start = time.time()
                    self.assign_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)
                    timer += time.time() - start

                if self.visualize:
                    self.plotter.plot_edge(node_near, node_new)
            k += 1

        index = self.search_goal_parent()
        self.goal = self.vertex[index]
        self.path = self.extract_path(self.goal)[::-1]
        return self.path

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
        return actions

    def arctan_clipped(self, loc1, loc2):
        heading = np.arctan2(loc2[1] - loc1[1], loc2[0] - loc1[0])
        if heading < -np.pi:
            heading += 2 * np.pi
        elif heading > np.pi:
            heading -= 2 * np.pi
        return heading

    def predict(self, observation, goal='closest', deterministic=True, plot=False):
        self.observations.process_observation(observation)
        if len(self.actions) == 0:
            closest_known_hideout, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
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
                plt.savefig('figures/path_rrt_star.png', dpi=600)
                plt.show()
            self.actions = self.convert_path_to_actions(path)
        return [self.actions.pop(0)]

class RRTStarAdversarialAvoid(RRTStarAdversarial):
    def __init__(env,             
            n_iter=1000, 
            step_len=150, 
            search_radius=150, 
            max_speed=15,
            terrain_cost_coef=500, 
            visualize=False,
            gamma=15, 
            goal_sample_rate=0.1):

        super().__init__(env, n_iter, step_len, search_radius, max_speed, terrain_cost_coef, visualize, gamma, goal_sample_rate)
    
    def predict(self, observation, deterministic=True, plot=False):
        self.observations.process_observation(observation)
        if len(self.actions) == 0:
            closest_known_hideout, closest_unknown_hideout = self.get_closest_hideouts(self.observations.location)
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
                plt.savefig('figures/path_rrt_star.png', dpi=600)
                plt.show()
            self.actions = self.convert_path_to_actions(path)
        return [self.actions.pop(0)]

class Plotter:
    def __init__(self, terrain, startpos, endpos, plot_mountains=True, live=True):
        self.terrain = terrain
        self.startpos = startpos
        self.endpos = endpos
        self.plot_mountains = plot_mountains
        self.initialize_plot()
        self.live = live

    def initialize_plot(self):
        fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, 2428])
        self.ax.set_ylim([0, 2428])
        self.ax.set_aspect('equal')
        # self.ax.scatter(self.terrain.mountain_locations[0][1], self.terrain.mountain_locations[0][0], marker='H', color='red')
        # self.ax.scatter(self.terrain.mountain_locations[1][1], self.terrain.mountain_locations[1][0], marker='H', color='red')
        # self.ax.imshow()
        self.ax.scatter(self.startpos[0], self.startpos[1], marker='o', s=200, color='blue', zorder=3)
        self.ax.scatter(self.endpos[0], self.endpos[1], marker='X', s=200, color='blue', zorder=3)
        
        if self.plot_mountains:
            for mountain_location in self.terrain.mountain_locations:
                cir = plt.Circle((mountain_location[1], mountain_location[0]), MOUNTAIN_OUTER_RANGE, color='blue')
                self.ax.add_patch(cir)

    def plot_edge(self, node1, node2, color='red', linewidth=1, zorder=1):
        xs = [node1.x, node2.x]
        ys = [node1.y, node2.y]
        self.ax.plot(xs, ys, color=color, zorder=zorder, linewidth=linewidth)
        if self.live:
            plt.pause(0.0001)

    def create_graph(self, node_root, background_img=None):
        if background_img is not None:
            self.ax.imshow(background_img)

        OPEN = QueueFIFO()
        OPEN.put(node_root)

        while not OPEN.empty():
            node = OPEN.get()

            if len(node.children) == 0:
                continue

            for node_c in node.children:
                self.plot_edge(node, node_c, color=c_str['orange'], zorder=1)
                OPEN.put(node_c)

    def create_path_node(self, node_end):
        node = node_end
        while node.parent is not None:
            self.plot_edge(node.parent, node, color=c_str['red'], linewidth=3, zorder=2)
            self.ax.scatter(node.x, node.y, marker='o', s=40, color=c_str['red'], zorder=2)
            node = node.parent

    def create_path(self, path):
        window_size = 2
        for i in range(len(path) - window_size + 1):
            start, end = path[i: i + window_size]
            xs = [start[0], end[0]]
            ys = [start[1], end[1]]
            self.ax.plot(xs, ys, color=c_str['red'], linewidth=3, zorder=2)
            self.ax.scatter(start[0], start[1], marker='o', s=40, color=c_str['red'], zorder=2)
