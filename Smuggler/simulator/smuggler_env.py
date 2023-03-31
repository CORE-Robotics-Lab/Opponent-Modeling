import copy
import math
from types import SimpleNamespace

import gc
import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

from PIL import Image
from dataclasses import dataclass
from gym import spaces
from tqdm import tqdm
from enum import Enum, auto
from numpy import genfromtxt

from simulator.abstract_object import AbstractObject
from simulator.camera import Camera
from simulator.fugitive import Fugitive
from simulator.helicopter import Helicopter
from simulator.hideout import Hideout
from simulator.rendezvous import Rendezvous
from simulator.search_party import SearchParty
from simulator.terrain import Terrain
from simulator.utils import create_camera_net

from simulator.observation_spaces import create_observation_space_ground_truth, create_observation_space_fugitive, \
    create_observation_space_blue_team, create_observation_space_prediction
from simulator.observation_spaces import transform_blue_detection_of_fugitive


class ObservationType(Enum):
    Fugitive = auto()
    FugitiveGoal = auto()
    Blue = auto()
    GroundTruth = auto()
    Prediction = auto()


@dataclass
class RewardScheme:
    time: float = -5e-4
    known_detected: float = -1.
    known_undetected: float = 1.
    unknown_detected: float = -2.
    unknown_undetected: float = 2.
    timeout: float = -3.


presets = RewardScheme.presets = SimpleNamespace()
presets.default = RewardScheme()
presets.none = RewardScheme(0., 0., 0., 0., 0., 0.)
presets.any_hideout = RewardScheme(known_detected=1., known_undetected=1., unknown_detected=1., unknown_undetected=1.,
                                   timeout=-1.)
presets.time_only = copy.copy(presets.none)
presets.time_only.time = presets.default.time
presets.timeout_only = copy.copy(presets.none)
presets.timeout_only.timeout = -3.
del presets  # access as RewardScheme.presets


class SmugglerBothEnv(gym.Env):
    """
    SmugglerEnv simulates the smuggler behavior in a grid world.
    The considered factors include
        - smuggler
        - rendezvous points
        - hideouts (targets)
        - max-time (72 hours)
        - helicopters
        - cutters

    *Detection is encoded by a three tuple [b, x, y] where b in binary. If b=1 (detected), [x, y] will have the detected location in world coordinates. If b=0 (not detected), [x, y] will be [-1, -1].

    State space
        - Time
        - Locations of [cameras, helicopters, helicopter dropped cameras, hideouts, search parties, fugitive]
        - Detection of the fugitive from [cameras, helicopters, helicopter dropped cameras, search parties]
        - Fugitive's detection of [helicopters, helicopter dropped cameras, search parties]

    Observation space (fugitive)
        - Time
        - Locations of [known cameras, hideouts]
        - Self location, speed, heading
        - Detection of [helicopters, helicopter dropped cameras, search parties]

    Action space
        - 2 dimensional: speed [1,15] x direction [-pi, pi]

    Observation space (good guys')
        - Time
        - Locations of [cameras, helicopters, helicopter dropped cameras, search parties, known hideouts]
        - Detection of the fugitive from [cameras, helicopters, helicopter dropped cameras, search parties]
        - Terrain

    Coordinate system:
        - By default, we use longitude and latitude coordinates:
        lat
        ^
        |
        |
        |
        |
        |----------->long

    Limitations:
        - Food and towns are not implemented yet
        - Sprint time maximum is not implemented yet. However, sprinting does still have drawbacks (easier to be detected)
        - No rain/fog
        - Fixed terrain, fixed hideout locations
        - Detection does not utilize an error ellipse. However, detection still has the range-based PoD.
        - Helicopter dropped cameras are not implemented yet.

    Details:
        - Good guys mean the side of search parties (opposite of the fugitive)
        - Each grid represents 21 meters
        - Grid size is dependent on lat/long coordinates given
        - We have continuous speed profile from 1 grid/timestep to 15 grids/timestep (1.26km/h to 18.9km/h)
        - Right now we have by default:
            - 2 search parties
            - 1 helicopter
            - 3 rendezvous points
            - 
    """

    def __init__(self,
                 terrain=None,
                 terrain_map=None,
                 num_towns=0,
                 num_search_parties=2,
                 num_helicopters=2,
                 helicopter_battery_life=360,
                 helicopter_recharge_time=360,
                 spawn_mode='normal',
                 spawn_range=15.,
                 max_timesteps=4320,
                 hideout_radius=50.,
                 reward_scheme=None,
                 random_hideout_locations=False,
                 num_known_hideouts=0,
                 num_unknown_hideouts=2,
                 known_hideout_locations = [], # leftover from Fugitive
                 unknown_hideout_locations = [[4300, 3400], [5000, 3000], [6000, 2600], [7000, 2000]],
                 rendezvous_points = [[3000, 1500], [2000, 2500], [5000, 1000], [6300, 500]],
                 random_cameras=False,
                 min_distance_from_hideout_to_start=1000,
                 num_random_unknown_cameras=25,
                 num_random_known_cameras=25,
                 camera_net_bool=False,
                 camera_net_path=None,
                 camera_range_factor=1,
                 camera_file_path="simulator/camera_locations/original.txt",
                 observation_step_type="Fugitive",  # Fugitive, Blue, GroundTruth
                 observation_terrain_feature=True,
                 include_camera_at_start=False,
                 include_start_location_blue_obs=False,
                 step_reset=True,
                 debug=False,
                 store_last_k_fugitive_detections=False,
                 ratio_render = 0.15,
                 lat_indices = (70, 102),
                 long_indices = (215, 286),
                 prisoner_starting = [4000, 100],
                 capture_radius = 100, 
                 ):
        """
        PrisonerEnv simulates the prisoner behavior in a grid world.
        :param terrain: If given, the terrain is used from this object
        :param terrain_map_file: This is the file that contains the terrain map, only used if terrain is None
            If none, the default map is used.
            Currently all the maps are stored in "/star-data/prisoner-maps/"
                We load in the map from .npy file, we use csv_generator.py to convert .nc to .npy
            If directory, cycle through all the files upon reset
            If single .npy file, use that file
        :param num_towns:
        :param num_search_parties:
        :param num_helicopters:
        :param random_hideout_locations: If True, hideouts are placed randomly with num_known_hideouts and num_unknown_hideouts
            If False, hideouts are selected from known_hideout_locations and unknown_hideout_locations based on the num_known_hideouts and num_unknown_hideouts
        :param num_known_hideouts: number of hideouts known to good guys
        :param num_unknown_hideouts: hideouts unknown to the good guys
        :param: known_hideout_locations: locations of known hideouts when random_hideout_locations=False
        :param: unknown_hideout_locations: locations of unknown hideouts when random_hideout_locations=False
        :param helicopter_battery_life: how many minutes the helicopter can last in the game
        :param helicopter_recharge_time: how many minutes the helicopter need to recharge itself
        :param spawn_mode: how the prisoner location is initialized on reset. Can be:
            'normal': the prisoner is spawned in the northeast corner
            'uniform': the prisoner spawns at a uniformly sampled random location
            'uniform_hideout_dist': spawn the prisoner at min_distance_from_hideout_to_start from the hideouts
                        This assumes the hideouts are chosen first
            'hideout': the prisoner spawns within `spawn_range` of the hideout
        :param spawn_range: how far from the edge of the hideout the prisoner spawns in 'hideout' mode, or how far from the corner the prisoner spawn in 'corner' mode
        :param max_timesteps: time horizon for each rollout. Default is 4320 (minutes = 72 hours)
        :param hideout_radius: minimum distance from a hideout to be considered "on" the hideout
        :param reward_scheme: a RewardScheme object definining reward scales for different events. If omitted, a default will be used. A custom one can be constructed. Several presets are available under RewardScheme.presets.
        :param known_hideout_locations: list of tuples of known hideout locations
        :param unknown_hideout_locations: list of tuples of unknown hideout locations
        :param random_cameras: boolean of whether to use random camera placements or fixed camera placements
        :param num_random_unknown_cameras: number of random unknown cameras
        :param num_random_known_cameras: number of random known cameras
        :param camera_file_path: path to the file containing the camera locations for the unknown cameras. This it for us to test the Filtering algorithm
        :param camera_net_bool: boolean of whether to use the camera net around the fugitive or not
        :param camera_net_path: if None, place camera net by generating, if path, use the path
        :observation_step_type: What observation is returned in the "step" and "reset" functions
            'Fugitive': Returns fugitive observations
            'Blue': Returns observations from the BlueTeam (aka blue team's vision of the fugitive)
            'GroundTruth': Returns information of all agents in the environment
            'Prediction': Returns fugitive observations but without the unknown hideouts
        :observation_terrain_feature: boolean of whether to include the terrain feature in the observation
        :step_reset: boolean of whether to reset the game after the episode is over or just wait at the final location no matter what action is given to it
            This is to make the multi-step prediction rollouts to work properly.
            Default is True 
        :param include_start_location_blue_obs: boolean of whether to include the start location of the prisoner in the blue team observation
            Default is True
        :param store_last_k_fugitive_detections: Whether or not to store the last k(=8) detections of the fugitive
        """

        self.prisoner_start_location = prisoner_starting
        # self.terrain_list = []
        self.DEBUG = debug
        self.store_last_k_fugitive_detections = store_last_k_fugitive_detections
        forest_color_scale = 10
        self.ratio_render = ratio_render
        self.terrain = Terrain(forest_color_scale=forest_color_scale, lat_indices = lat_indices, long_indices = long_indices)

        self.x_dim_render = int(self.terrain.dim_x * self.ratio_render)
        self.y_dim_render = int(self.terrain.dim_y * self.ratio_render)

        # self._cached_terrain_images = [terrain.visualize(self.x_dim_render, self.y_dim_render, just_matrix=True) for terrain in self.terrain_list]
        self.terrain.cache_images(self.x_dim_render, self.y_dim_render, 20)
        self._cached_terrain_embeddings = [np.array([])]
        terrain_embedding_size = 0

        # initialize terrain for this run
        # self.set_terrain_paramaters()
        self.x_render_scale = self.x_dim_render / self.terrain.dim_x 
        self.y_render_scale = self.y_dim_render / self.terrain.dim_y
        self.prisoner = Fugitive(self.terrain, prisoner_starting)  # the actual spawning will happen in set_up_world

        # Read in the cameras from file
        if random_cameras: 
            self.num_random_unknown_cameras = num_random_unknown_cameras
            self.num_random_known_cameras = num_random_known_cameras 
        else:
            self.camera_file_path = camera_file_path
            self.known_camera_locations, self.unknown_camera_locations = self.read_camera_file(camera_file_path)

        self.include_camera_at_start = include_camera_at_start
        
        self.dim_x = self.terrain.dim_x
        self.dim_y = self.terrain.dim_y

        # self.num_unknown_cameras = self.num_unknown_cameras
        self.num_towns = num_towns
        self.num_search_parties = num_search_parties
        self.num_helicopters = num_helicopters
        self.random_hideout_locations = random_hideout_locations

        self.num_known_hideouts = num_known_hideouts
        self.num_unknown_hideouts = num_unknown_hideouts
        # self.num_known_cameras = self.num_known_cameras + num_known_hideouts # add camera for each known hideout

        self.helicopter_battery_life = helicopter_battery_life
        self.helicopter_recharge_time = helicopter_recharge_time
        self.spawn_mode = spawn_mode
        self.spawn_range = spawn_range
        self.hideout_radius = hideout_radius
        self.reward_scheme = reward_scheme or RewardScheme()  # accept a custom or use the default
        self.known_hideout_locations = known_hideout_locations
        self.unknown_hideout_locations = unknown_hideout_locations
        self.random_cameras = random_cameras
        self.camera_file_path = camera_file_path
        self.camera_range_factor = camera_range_factor
        self.current_prisoner_speed = 3  # initialize prisoner speed, used to render detection range
        self.step_reset = step_reset
        self.camera_net_bool = camera_net_bool
        self.camera_net_path = camera_net_path
        self.include_start_location_blue_obs = include_start_location_blue_obs
        self.min_distance_from_hideout_to_start = min_distance_from_hideout_to_start
        self.rendezvous_points = rendezvous_points
        self.num_rendezvous_points = len(rendezvous_points)
        self.capture_radius = capture_radius

        self.max_timesteps = max_timesteps  # 72 hours = 4320 minutes

        assert isinstance(self.reward_scheme, (type(None), str, RewardScheme))
        if isinstance(self.reward_scheme, str):
            self.reward_scheme = getattr(RewardScheme.presets, self.reward_scheme)

        self.action_space = spaces.Box(low=np.array([0, -np.pi]), high=np.array([15, np.pi]))

        # initialization of variables
        self.camera_list = []
        self.helicopters_list = []
        self.hideout_list = []
        self.search_parties_list = []
        self.town_list = []
        self.timesteps = 0
        self.done = False
        self.is_detected = False
        self.last_detected_timestep = 0

        # initialize objects
        self.set_up_world()

        self.blue_observation_space, self.blue_obs_names = create_observation_space_blue_team(
                                        num_known_cameras=self.num_known_cameras, 
                                        num_unknown_cameras=self.num_unknown_cameras, 
                                        num_known_hideouts=self.num_known_hideouts,
                                        num_helicopters=self.num_helicopters, 
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size,
                                        include_start_location_blue_obs=include_start_location_blue_obs)
        self.fugitive_observation_space, self.fugitive_obs_names = create_observation_space_fugitive(num_known_cameras=self.num_known_cameras, 
                                        num_known_hideouts=self.num_known_hideouts, 
                                        num_unknown_hideouts=self.num_unknown_hideouts, 
                                        num_helicopters=self.num_helicopters, 
                                        num_search_parties=self.num_search_parties,
                                        num_rendezvous_points = self.num_rendezvous_points,
                                        terrain_size=terrain_embedding_size)

        self.gt_observation_space, self.gt_obs_names = create_observation_space_ground_truth(num_known_cameras=self.num_known_cameras, 
                                        num_unknown_cameras=self.num_unknown_cameras, 
                                        num_known_hideouts=self.num_known_hideouts, 
                                        num_unknown_hideouts=self.num_unknown_hideouts,
                                        num_helicopters=self.num_helicopters, 
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size)

        self.prediction_observation_space, self.prediction_obs_names = create_observation_space_prediction(num_known_cameras=self.num_known_cameras,
                                        num_known_hideouts=self.num_known_hideouts,
                                        num_helicopters=self.num_helicopters,
                                        num_search_parties=self.num_search_parties,
                                        terrain_size=terrain_embedding_size)

        self.prisoner_location_history = [self.prisoner.location.copy()]

        # load image assets
        self.known_camera_pic = Image.open("simulator/assets/camera_blue.png")
        self.unknown_camera_pic = Image.open("simulator/assets/camera_red.png")
        self.known_hideout_pic = Image.open("simulator/assets/star.png")
        self.unknown_hideout_pic = Image.open("simulator/assets/star_blue.png")
        self.search_party_pic = Image.open("simulator/assets/boat_white.png")
        self.helicopter_pic = Image.open("simulator/assets/helicopter.png")
        self.prisoner_pic = Image.open("simulator/assets/prisoner.png")
        self.detected_prisoner_pic = Image.open("simulator/assets/detected_prisoner.png")

        self.known_camera_pic_cv = cv2.imread("simulator/assets/camera_blue.png", cv2.IMREAD_UNCHANGED)
        self.unknown_camera_pic_cv = cv2.imread("simulator/assets/camera_red.png", cv2.IMREAD_UNCHANGED)
        self.known_hideout_pic_cv = cv2.imread("simulator/assets/star.png", cv2.IMREAD_UNCHANGED)
        self.unknown_hideout_pic_cv = cv2.imread("simulator/assets/star_green.png", cv2.IMREAD_UNCHANGED)
        self.search_party_pic_cv = cv2.imread("simulator/assets/boat_white.png", cv2.IMREAD_UNCHANGED)
        self.helicopter_pic_cv = cv2.imread("simulator/assets/airplane.png", cv2.IMREAD_UNCHANGED)
        self.helicopter_no_pic_cv = cv2.imread("simulator/assets/airplane_no.png", cv2.IMREAD_UNCHANGED)
        self.prisoner_pic_cv = cv2.imread("simulator/assets/smuggler.png", cv2.IMREAD_UNCHANGED)
        self.detected_prisoner_pic_cv = cv2.imread("simulator/assets/smuggler_detected.png", cv2.IMREAD_UNCHANGED)
        self.rendezvous_pic_cv = cv2.imread("simulator/assets/map_marker.png", cv2.IMREAD_UNCHANGED)
        self.rendezvous_pic_check_cv = cv2.imread("simulator/assets/map_marker_check.png", cv2.IMREAD_UNCHANGED)

        self.default_asset_size = 25
        # Store (t,x,y) for last k detections. Only updated if store_last_k_fugitive_detections is True
        self.last_k_fugitive_detections = [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1],
                                           [-1, -1, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
                    
        # self.render(show=True)

    def read_camera_file(self, camera_file_path):
        """Generate a lists of camera objects from file

        Args:
            camera_file_path (str): path to camera file

        Raises:
            ValueError: If Camera file does not have a u or k at beginning of each line

        Returns:
            (list, list): Returns known camera locations and unknown camera locations
        """
        unknown_camera_locations = []
        known_camera_locations = []
        camera_file = open(camera_file_path, "r").readlines()
        for line in camera_file:
            line = line.strip().split(",")
            if line[0] == 'u':
                unknown_camera_locations.append([int(line[1]), int(line[2])])
            elif line[0] == 'k':
                known_camera_locations.append([int(line[1]), int(line[2])])
            else:
                raise ValueError(
                    "Camera file format is incorrect, each line must start with 'u' or 'k' to denote unknown or known")
        return known_camera_locations, unknown_camera_locations


    def place_random_hideouts(self):
        for known_hid in range(self.num_known_hideouts):
            if known_hid == 0:
                location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                while np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location) < self.min_distance_from_hideout_to_start:
                    location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                if self.DEBUG:
                    print('prisoner location: ', self.prisoner.location)
                    print('hideout location: ', location)
                    print('distance: ', np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location))
                self.hideout_list.append(Hideout(self.terrain, location=location, known_to_good_guys=True))
            else:
                location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                # make sure hideout is far from each other and far from start location
                s = [tuple(i.location) for i in self.hideout_list]
                dists = np.array([math.sqrt((location[0] - s0) ** 2 + (location[1] - s1) ** 2) for s0, s1 in s])
                while np.linalg.norm(np.array([location[0], location[
                    1]]) - self.prisoner.location) <= self.min_distance_from_hideout_to_start \
                        or (dists < self.min_distance_between_hideouts).any():
                    location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                    s = [tuple(i.location) for i in self.hideout_list]
                    dists = np.array([math.sqrt((location[0] - s0) ** 2 + (location[1] - s1) ** 2) for s0, s1 in s])
                if self.DEBUG:
                    print('prisoner location: ', self.prisoner.location)
                    print('hideout location: ', location)
                    print('distance: ', np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location))
                self.hideout_list.append(Hideout(self.terrain, location=location, known_to_good_guys=True))

        for unknown_hid in range(self.num_unknown_hideouts):
            if len(self.hideout_list) >=1:
                # make sure hideout is far from each other and far from start location
                location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                s = [tuple(i.location) for i in self.hideout_list]
                dists = np.array([math.sqrt((location[0] - s0) ** 2 + (location[1] - s1) ** 2) for s0, s1 in s])
                while np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location) <= self.min_distance_from_hideout_to_start\
                        or (dists < self.min_distance_between_hideouts).any():
                    location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                    s = [tuple(i.location) for i in self.hideout_list]
                    dists = np.array([math.sqrt((location[0] - s0) ** 2 + (location[1] - s1) ** 2) for s0, s1 in s])
                    if self.DEBUG:
                        print('prisoner location: ', self.prisoner.location)
                        print('hideout location: ', location)
                        print('distance: ', np.linalg.norm(np.array([location[0], location[1]]) - self.prisoner.location))
                self.hideout_list.append(Hideout(self.terrain, location=location, known_to_good_guys=False))
            else:
                location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
                self.hideout_list.append(Hideout(self.terrain, location=location, known_to_good_guys=False))

    def place_fixed_hideouts(self):
        # specify hideouts' locations. These are passed in from the input args
        # We select a number of hideouts from num_known_hideouts and num_unknown_hideouts

        assert self.num_known_hideouts <= len(self.known_hideout_locations), f"Must provide a list of known_hideout_locations ({len(self.known_hideout_locations)}) greater than number of known hideouts {self.num_known_hideouts}"
        assert self.num_unknown_hideouts <= len(self.unknown_hideout_locations), f"Must provide a list of known_hideout_locations ({len(self.unknown_hideout_locations)}) greater than number of known hideouts {self.num_unknown_hideouts}"

        known_hideouts = random.sample(self.known_hideout_locations, self.num_known_hideouts)
        unknown_hideouts = random.sample(self.unknown_hideout_locations, self.num_unknown_hideouts)

        self.hideout_list = []
        for hideout_location in known_hideouts:
            self.hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=True))

        for hideout_location in unknown_hideouts:
            self.hideout_list.append(Hideout(self.terrain, location=hideout_location, known_to_good_guys=False))

    def set_up_world(self):
        """
        This function places all the objects,
        Right now,
            - cameras are initialized randomly
            - helicopter is initialized randomly
            - hideouts are initialized always at [20, 80], [100, 20]
            - search parties are initialized randomly
            - prisoner is initialized by different self.spawn_mode
        """
        self.camera_list = []
        self.helicopters_list = []
        self.hideout_list = []
        self.search_parties_list = []
        self.town_list = []
        self.hideout_list = []
        self.min_distance_between_hideouts = 300
        

        # randomized
        if not self.random_hideout_locations:
            self.place_fixed_hideouts()
        else:
            raise NotImplementedError
            # Random hideouts have not been implemented with spawn mode as uniform hideout dist
            # Random hideouts need to have prisoner location initialized first
            self.place_random_hideouts()

        self.rendezvous_point_list = []
        for rendezvous_point in self.rendezvous_points:
            self.rendezvous_point_list.append(Rendezvous(self.terrain, location=rendezvous_point))
        self.reached_rendezvous = False
        self.rendezvous_target_index = random.randint(0, len(self.rendezvous_point_list)-1)

        # prisoner_location = [0, 4400]
        if self.spawn_mode == 'normal':
            # prisoner_location =  [4400, 4400]
            prisoner_location = self.prisoner_start_location
        elif self.spawn_mode == 'random':
            # generate the fugitive randomly near the top right corner
            prisoner_location = self.terrain.generate_random_location_open_terrain()
        else:
            raise ValueError('Unknown spawn mode "%s"' % self.spawn_mode)

        
        self.prisoner = Fugitive(self.terrain, prisoner_location)
        self.prisoner_start_location = prisoner_location

        # specify cameras' initial locations
        if(self.random_cameras):
            # randomized 
            known_camera_locations = [AbstractObject.generate_random_locations(self.dim_x, self.dim_y) for _ in range(self.num_random_known_cameras)]
            unknown_camera_locations = [AbstractObject.generate_random_locations(self.dim_x, self.dim_y) for _ in range(self.num_random_unknown_cameras)]
        else:
            known_camera_locations = self.known_camera_locations[:]
            unknown_camera_locations = copy.deepcopy(self.unknown_camera_locations)
        
        if self.camera_net_bool:
            if self.camera_net_path is None:
                cam_locs = create_camera_net(prisoner_location, 
                                        dist_x=360, 
                                        dist_y=360, 
                                        spacing=30, 
                                        include_camera_at_start=self.include_camera_at_start,
                                        board_size=(self.dim_x, self.dim_y))
                unknown_camera_locations.extend(cam_locs.tolist())
            else:
                known_net, unknown_net = self.read_camera_file(self.camera_net_path)
                known_camera_locations.extend(known_net)
                unknown_camera_locations.extend(unknown_net)

        # append cameras at known hideouts
        for i in self.hideout_list:
            if i.known_to_good_guys:
                known_camera_locations.append(i.location)

        # initialize these variables for observation spaces
        self.num_known_cameras = len(known_camera_locations)
        self.num_unknown_cameras = len(unknown_camera_locations)

        for counter in range(self.num_known_cameras):
            camera_location = known_camera_locations[counter]
            self.camera_list.append(Camera(self.terrain, camera_location, known_to_fugitive=True))

        for counter in range(self.num_unknown_cameras):
            camera_location = unknown_camera_locations[counter]
            self.camera_list.append(Camera(self.terrain, camera_location, known_to_fugitive=False,
                                           detection_object_type_coefficient=self.camera_range_factor))

        # specify helicopters' initial locations
        for _ in range(self.num_helicopters):
            # helicopter_location = AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
            helicopter_location = self.terrain.generate_random_location_open_terrain()
            self.helicopters_list.append(
                Helicopter(self.terrain, helicopter_location, speed=127))  # 100mph=127 grids/timestep

        search_party_initial_locations = []
        for _ in range(self.num_search_parties):
            search_party_init_loc = self.terrain.generate_random_location_open_terrain()
            # search_party_initial_locations.append(AbstractObject.generate_random_locations(self.dim_x, self.dim_y))
            search_party_initial_locations.append(search_party_init_loc)

        # generate search party lists
        for counter in range(self.num_search_parties):
            search_party_location = search_party_initial_locations[
                counter]  # AbstractObject.generate_random_locations(self.dim_x, self.dim_y)
            self.search_parties_list.append(SearchParty(self.terrain, search_party_location, speed=6.5))  # speed=4

    @property
    def hideout_locations(self):
        return [hideout.location for hideout in self.hideout_list]

    def get_state(self):
        """
        Compile a dictionary to represent environment's current state (only including things that will change in .step())
        :return: a dictionary with prisoner_location, search_party_locations, helicopter_locations, timestep, done, prisoner_location_history, is_detected
        """
        prisoner_location = self.prisoner.location.copy()
        search_party_locations = []
        for search_party in self.search_parties_list:
            search_party_locations.append(search_party.location.copy())
        helicopter_locations = []
        for helicopter in self.helicopters_list:
            helicopter_locations.append(helicopter.location.copy())
        timestep = self.timesteps
        done = self.done
        prisoner_location_history = self.prisoner_location_history.copy()
        is_detected = self.is_detected

        prediction_observation = self._prediction_observation.copy()
        fugitive_observation = self._fugitive_observation.copy()
        ground_truth_observation = self._ground_truth_observation.copy()
        blue_observation = self._blue_observation.copy()

        # print(self.search_parties_list[0].location)
        return {
            "prisoner_location": prisoner_location,
            "search_party_locations": search_party_locations,
            "helicopter_locations": helicopter_locations,
            "timestep": timestep,
            "done": done,
            "prisoner_location_history": prisoner_location_history,
            "is_detected": is_detected,
            # "blue_heuristic": copy.deepcopy(self.blue_heuristic),
            "prediction_observation": prediction_observation,
            "fugitive_observation": fugitive_observation,
            "ground_truth_observation": ground_truth_observation,
            "blue_observation": blue_observation,
            "done": self.done
        }

    def set_state(self, state_dict):
        """
        Set the state of the env by state_dict. Paired with `get_state`
        :param state_dict: a state dict returned by `get_state`
        """
        self.prisoner.location = state_dict["prisoner_location"].copy()
        for i, search_party in enumerate(self.search_parties_list):
            search_party.location = state_dict["search_party_locations"][i].copy()
        for i, helicopter in enumerate(self.helicopters_list):
            helicopter.location = state_dict["helicopter_locations"][i].copy()
        self.timesteps = state_dict["timestep"]
        self.done = state_dict["done"]
        self.prisoner_location_history = state_dict["prisoner_location_history"].copy()
        self.is_detected = state_dict["is_detected"]
        # self.blue_heuristic = state_dict["blue_heuristic"]

        # self.search_parties_list = self.blue_heuristic.search_parties
        # self.helicopters_list = self.blue_heuristic.helicopters

        # set previous observations
        self._prediction_observation = state_dict["prediction_observation"].copy()
        self._fugitive_observation = state_dict["fugitive_observation"].copy()
        self._ground_truth_observation = state_dict["ground_truth_observation"].copy()
        self._blue_observation = state_dict["blue_observation"].copy()
        self.done = state_dict["done"]
        gc.collect()
        # self.blue_heuristic.step(self.prisoner.location)

    def search_party_near_prisoner(self):
        """ Check if any search parties are within radius of prisoner for capture """
        for search_party in self.search_parties_list:
            if ((np.asarray(search_party.location) - np.asarray(
                    self.prisoner.location)) ** 2).sum() ** .5 <= self.capture_radius + 1e-6:
                return True
        return False

    def step_both(self, red_action: np.ndarray, blue_action: np.ndarray):
        """
        The environment moves one timestep forward with the action chosen by the agent.
        :param red_action: an speed and direction vector for the red agent
        :param blue_action: currently a triple of [dx, dy, speed] where dx and dy is the vector
            pointing to where the agent should go
            this vector should have a norm of 1
            we can potentially take np.arctan2(dy, dx) to match action space of fugitive


        :return: observation, reward, done (boolean), info (dict)
        """
        # print("Before step", self.search_parties_list[0].location)
        if self.done:
            if self.step_reset:
                raise RuntimeError("Episode is done")
            else:
                observation = np.zeros(self.observation_space.shape)
                total_reward = 0
                return observation, total_reward, self.done, {}
        assert self.action_space.contains(red_action), f"Actions should be in the action space, but got {red_action}"

        self.timesteps += 1

        if self.timesteps % 180 == 0:
            # change the terrain every n timesteps
            self.terrain.current_index += 1

        old_prisoner_location = self.prisoner.location.copy()

        # move red agent
        direction = np.array([np.cos(red_action[1]), np.sin(red_action[1])])

        fugitive_speed = red_action[0]
        self.current_prisoner_speed = fugitive_speed

        prisoner_location = np.array(self.prisoner.location, dtype=np.float)
        
        lat_long = self.terrain.convert_x_y_to_lat_long(prisoner_location)
        new_location = prisoner_location + direction * fugitive_speed

        
        # bump back from mountain
        if self.terrain.in_mountain(new_location):
            lat_long = self.terrain.convert_x_y_to_lat_long(new_location)
            new_location = np.array(old_prisoner_location)

        # finish moving the prisoner
        self.prisoner.location = new_location.tolist()
        self.prisoner_location_history.append(self.prisoner.location.copy())

        # move blue agents
        for i, search_party in enumerate(self.search_parties_list):
            # getattr(search_party, command)(*args, **kwargs)
            direction = blue_action[i][0:2]
            speed = blue_action[i][2]
            new_location = search_party.path_v3(direction=direction, speed=speed)
            # if self.terrain.violate_edge_constraints(new_location[0], new_location[1], 1, 1):
            
            lat_long = self.terrain.convert_x_y_to_lat_long(new_location)
            if lat_long[0] >= self.terrain.lat_dim or lat_long[0] <= 0 or lat_long[1] >= self.terrain.long_dim or lat_long[1] <= 0:
                new_location = np.clip(new_location, [0, 0], [self.terrain.dim_x-1, self.terrain.dim_y-1])
            
            if not self.terrain.in_mountain(new_location):
                search_party.location = new_location
        
        if self.is_helicopter_operating():
            for j, helicopter in enumerate(self.helicopters_list):
                # getattr(helicopter, command)(*args, **kwargs)
                direction = blue_action[i + j + 1][0:2]
                speed = blue_action[i + j + 1][2]
                new_location = helicopter.path_v3(direction=direction, speed=speed)
                lat_long = self.terrain.convert_x_y_to_lat_long(new_location)
                if lat_long[0] >= self.terrain.lat_dim or lat_long[0] <= 0 or lat_long[1] >= self.terrain.long_dim or lat_long[1] <= 0:
                    new_location = np.clip(new_location, [0, 0], [self.terrain.dim_x-1, self.terrain.dim_y-1])
                if not self.terrain.in_mountain(new_location):
                    helicopter.location = new_location

        near_rendezvous = self.near_rendezvous_point()
        if near_rendezvous is not None:
            near_rendezvous.reached = True
            self.reached_rendezvous = True

        if self.search_party_near_prisoner():
            self.done = True

        if self.near_hideout() and self.reached_rendezvous is True:
            self.done = True

        # game ends?
        if self.timesteps >= self.max_timesteps:
            self.done = True

        # Construct observation from these
        parties_detection_of_fugitive = self._determine_blue_detection_of_red(fugitive_speed)
        fugitive_detection_of_parties = self._determine_red_detection_of_blue(fugitive_speed)
        self._fugitive_observation = self._construct_fugitive_observation(red_action, fugitive_detection_of_parties)
        self._prediction_observation = self._construct_prediction_observation(red_action, fugitive_detection_of_parties)
        self._ground_truth_observation = self._construct_ground_truth(red_action, fugitive_detection_of_parties,
                                                                      parties_detection_of_fugitive)

        parties_detection_of_fugitive_one_hot = transform_blue_detection_of_fugitive(parties_detection_of_fugitive)

        self._blue_observation = self._construct_blue_observation(parties_detection_of_fugitive_one_hot, self.include_start_location_blue_obs)

        # calculate reward
        self.is_detected = self.is_fugitive_detected(parties_detection_of_fugitive)
        if self.is_detected and self.store_last_k_fugitive_detections:
            self.last_k_fugitive_detections.pop(0)  # Remove old detection
            self.last_k_fugitive_detections.append([self.timesteps / self.max_timesteps,
                                                    self.prisoner.location[0] / self.dim_x,
                                                    self.prisoner.location[1] / self.dim_y])  # Append latest detection
        total_reward = self.get_reward()

        return self._fugitive_observation, self._blue_observation, total_reward, self.done, {}

    def step(self, red_action, blue_action):
        red_obs, blue_obs, total_reward, done, empty = self.step_both(red_action, blue_action)
        return red_obs, total_reward, done, empty

    @property
    def hideout_locations(self):
        return [hideout.location for hideout in self.hideout_list]

    def is_helicopter_operating(self):
        """
        Determines whether the helicopter is operating right now
        :return: Boolean indicating whether the helicopter is operating
        """
        timestep = self.timesteps % (self.helicopter_recharge_time + self.helicopter_battery_life)
        if timestep < self.helicopter_battery_life:
            return True
        else:
            return False

    @property
    def spawn_point(self):
        return self.prisoner_location_history[0].copy()

    @staticmethod
    def is_fugitive_detected(parties_detection_of_fugitive):
        for e, i in enumerate(parties_detection_of_fugitive):
            if e % 3 == 0:
                if i == 1:
                    return True
        return False

    def get_reward(self):
        # TODO recode this so combinations of scenarios are possible per timestep
        if self.timesteps == self.max_timesteps:
            return self.reward_scheme.timeout  # running out of time is bad!
        hideout = self.near_hideout()
        if hideout is not None:
            if hideout.known_to_good_guys:
                if self.is_detected:
                    return self.reward_scheme.known_detected
                else:
                    return self.reward_scheme.known_undetected
            else:
                if self.is_detected:
                    return self.reward_scheme.unknown_detected
                else:
                    return self.reward_scheme.unknown_undetected

        # game is not done. Simple sparse timestep reward
        return self.reward_scheme.time

    def near_hideout(self):
        """If the prisoner is within range of a hideout, return it. Otherwise, return None."""
        for hideout in self.hideout_list:
            if ((np.asarray(hideout.location) - np.asarray(
                    self.prisoner.location)) ** 2).sum() ** .5 <= self.hideout_radius + 1e-6:
                # print(f"Reached a hideout that is {hideout.known_to_good_guys} known to good guys")
                return hideout
        return None

    def near_rendezvous_point(self):
        """If the prisoner is within range of a rendezvous, return it. Otherwise, return None."""
        for rendezvous in self.rendezvous_point_list:
            if ((np.asarray(rendezvous.location) - np.asarray(
                    self.prisoner.location)) ** 2).sum() ** .5 <= self.hideout_radius + 1e-6:
                # print(f"Reached a hideout that is {hideout.known_to_good_guys} known to good guys")
                return rendezvous
        return None

    def _determine_red_detection_of_blue(self, speed):
        fugitive_detection_of_parties = []
        SPRINT_SPEED_THRESHOLD = 8
        for helicopter in self.helicopters_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, -1, -1])
            else:
                fugitive_detection_of_parties.extend(self.prisoner.detect(helicopter.location, helicopter))
        for search_party in self.search_parties_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, -1, -1])
            else:
                fugitive_detection_of_parties.extend(self.prisoner.detect(search_party.location, search_party))
        return fugitive_detection_of_parties

    def _determine_blue_detection_of_red(self, speed):
        parties_detection_of_fugitive = []
        for camera in self.camera_list:
            parties_detection_of_fugitive.extend(camera.detect(self.prisoner.location, speed))
        for helicopter in self.helicopters_list:
            parties_detection_of_fugitive.extend(helicopter.detect(self.prisoner.location, speed))
        for search_party in self.search_parties_list:
            parties_detection_of_fugitive.extend(search_party.detect(self.prisoner.location, speed))

        if any(parties_detection_of_fugitive[::3]):
            self.last_detected_timestep = self.timesteps
        return parties_detection_of_fugitive

    def _determine_detection(self, speed):
        fugitive_detection_of_parties = []
        SPRINT_SPEED_THRESHOLD = 8
        for helicopter in self.helicopters_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, -1, -1])
            else:
                fugitive_detection_of_parties.extend(self.prisoner.detect(helicopter.location, helicopter))
        for search_party in self.search_parties_list:
            if speed > SPRINT_SPEED_THRESHOLD:
                # when sprinting the prisoner cannot detect anything
                fugitive_detection_of_parties.extend([0, -1, -1])
            else:
                fugitive_detection_of_parties.extend(self.prisoner.detect(search_party.location, search_party))
        parties_detection_of_fugitive = []
        for camera in self.camera_list:
            parties_detection_of_fugitive.extend(camera.detect(self.prisoner.location, speed))
        for helicopter in self.helicopters_list:
            parties_detection_of_fugitive.extend(helicopter.detect(self.prisoner.location, speed))
        for search_party in self.search_parties_list:
            parties_detection_of_fugitive.extend(search_party.detect(self.prisoner.location, speed))

        if any(parties_detection_of_fugitive[::3]):
            self.last_detected_timestep = self.timesteps

        return fugitive_detection_of_parties, parties_detection_of_fugitive

    def _construct_fugitive_observation(self, action, fugitive_detection_of_parties):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param action: the action taken by the fugitive at this timestep
        :param fugitive_detection_of_parties: a list encoding fugitive's detection of all parties
        :param terrain: the terrain feature of the environment
        :return: the observation from the prisoner perspective
        """

        # NOTE: When editing, make sure this corresponds to names/orders in obs_names, in constructor
        # Future: Assign these using obs_names instead? may be slower...
        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                observation.append(camera.location[0] / self.dim_x)
                observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            observation.append(float(hideout.known_to_good_guys))
            observation.append(hideout.location[0] / self.dim_x)
            observation.append(hideout.location[1] / self.dim_y)
        observation.append(self.prisoner.location[0] / self.dim_x)
        observation.append(self.prisoner.location[1] / self.dim_y)
        observation.append(action[0] / 15)
        observation.append(action[1] / np.pi)
        observation.extend(fugitive_detection_of_parties)

        target_rendezvous_point = self.rendezvous_point_list[self.rendezvous_target_index]
        observation.append(target_rendezvous_point.location[0] / self.dim_x)
        observation.append(target_rendezvous_point.location[1] / self.dim_y)
        observation = np.array(observation)

        # observation = np.concatenate((observation, terrain))

        return observation

    def _construct_prediction_observation(self, action, fugitive_detection_of_parties):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param action: the action taken by the fugitive at this timestep
        :param fugitive_detection_of_parties: a list encoding fugitive's detection of all parties
        :param terrain: the terrain feature of the environment
        :return: the observation from the prisoner perspective
        """
        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                observation.append(camera.location[0] / self.dim_x)
                observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                observation.append(hideout.location[0] / self.dim_x)
                observation.append(hideout.location[1] / self.dim_y)
        observation.append(self.prisoner.location[0] / self.dim_x)
        observation.append(self.prisoner.location[1] / self.dim_y)
        observation.append(action[0] / 15)
        observation.append(action[1] / np.pi)
        observation.extend(fugitive_detection_of_parties)
        observation = np.array(observation)
        # observation = np.concatenate((observation, terrain))

        return observation

    def _construct_ground_truth(self, action, fugitive_detection_of_parties, parties_detection_of_fugitive):
        """
        Construct ground truth state.
        """
        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            observation.append(camera.location[0] / self.dim_x)
            observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            observation.append(float(hideout.known_to_good_guys))
            observation.append(hideout.location[0] / self.dim_x)
            observation.append(hideout.location[1] / self.dim_y)
        observation.append(self.prisoner.location[0] / self.dim_x)
        observation.append(self.prisoner.location[1] / self.dim_y)
        observation.append(action[0] / 15)
        observation.append(action[1] / np.pi)
        for helicopter in self.helicopters_list:
            observation.append(helicopter.location[0] / self.dim_x)
            observation.append(helicopter.location[1] / self.dim_y)
        for search_party in self.search_parties_list:
            observation.append(search_party.location[0] / self.dim_x)
            observation.append(search_party.location[1] / self.dim_y)

        # We include here the observations of each of the parties (don't need the locations given from these)
        for i in range(0, len(fugitive_detection_of_parties), 3):
            observation.append(fugitive_detection_of_parties[i])

        for i in range(0, len(parties_detection_of_fugitive), 3):
            observation.append(parties_detection_of_fugitive[i])

        observation = np.array(observation)
        # observation = np.concatenate((observation, terrain))
        return observation

    def _construct_blue_observation(self, parties_detection_of_fugitive, include_start_location_blue_obs=False):
        """
        Construct observation feature map from current states. For more info about the two parameters, check `self.step()`
        :param parties_detection_of_fugitive: a list encoding parties detection of the fugitive
        :return: the observation from the good guys perspective
        """

        observation = [self.timesteps / self.max_timesteps]
        for camera in self.camera_list:
            observation.append(camera.location[0] / self.dim_x)
            observation.append(camera.location[1] / self.dim_y)
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                observation.append(hideout.location[0] / self.dim_x)
                observation.append(hideout.location[1] / self.dim_y)
        for helicopter in self.helicopters_list:
            observation.append(helicopter.location[0] / self.dim_x)
            observation.append(helicopter.location[1] / self.dim_y)
        for search_party in self.search_parties_list:
            observation.append(search_party.location[0] / self.dim_x)
            observation.append(search_party.location[1] / self.dim_y)

        observation.extend(parties_detection_of_fugitive)

        if include_start_location_blue_obs:
            observation.append(self.prisoner_start_location[0]/ self.dim_x)
            observation.append(self.prisoner_start_location[1] / self.dim_y)

        observation = np.array(observation)
        # observation = np.concatenate((observation, terrain))

        return observation

    def cell_to_obs(self, cell):
        """
        Map a grid cell to the coordinates emitted in observations
        :param cell: integer sequence of length 2 within the range [(0, 0), (dim_x, dim_y))
        :return: np.ndarray of shape (2,) in the range [0, 1) of type np.float32
        """
        return np.array([cell[0] / self.dim_x, cell[1] / self.dim_y], dtype=np.float32)

    def obs_to_cell(self, coord):
        """
        Map a float coordinate in the observation space to the grid cell it most closely represents
        :param coord: float sequence of length 2 in the range [0, 1)
        :return: np.ndarray of shape (2,) in the range [(0, 0), (dim_x, dim_y))
        """
        return np.array([coord[0] * self.dim_x, coord[1] * self.dim_y], dtype=np.int)

    def reset(self, seed=None):
        """
        Reset the environment. Should be called whenever done==True
        :return: observation
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # self.set_terrain_paramaters()
        self.terrain.current_index = 0
        self.prisoner = Fugitive(self.terrain, [2400, 2400])  # the actual spawning will happen in set_up_world
        # Randomize the terrain

        self.timesteps = 0
        self.last_detected_timestep = 0
        self.done = False

        self.set_up_world()
        fugitive_detection_of_parties, parties_detection_of_fugitive = self._determine_detection(0.0)

        self.prisoner_location_history = [self.prisoner.location.copy()]
        self._fugitive_observation = self._construct_fugitive_observation([0.0, 0.0], fugitive_detection_of_parties)
        self._prediction_observation = self._construct_prediction_observation([0.0, 0.0], fugitive_detection_of_parties)
        self._ground_truth_observation = self._construct_ground_truth([0.0, 0.0], fugitive_detection_of_parties,
                                                                      parties_detection_of_fugitive)
        parties_detection_of_fugitive = transform_blue_detection_of_fugitive(parties_detection_of_fugitive)
        self._blue_observation = self._construct_blue_observation(parties_detection_of_fugitive,
                                                                self.include_start_location_blue_obs)

        assert self._blue_observation.shape == self.blue_observation_space.shape, "Wrong observation shape %s, %s" % (
        self._blue_observation.shape, self.blue_observation_space.shape)
        assert self._ground_truth_observation.shape == self.gt_observation_space.shape, "Wrong observation shape %s, %s" % (
        self._ground_truth_observation.shape, self.gt_observation_space.shape)
        assert self._fugitive_observation.shape == self.fugitive_observation_space.shape, "Wrong observation shape %s, %s" % (
        self._fugitive_observation.shape, self.fugitive_observation_space.shape)
        assert self._prediction_observation.shape == self.prediction_observation_space.shape, "Wrong observation shape %s, %s" % (
        self._fugitive_observation.shape, self.fugitive_observation_space.shape)

        return self._fugitive_observation

    def get_prediction_observation(self):
        return self._prediction_observation

    def get_fugitive_observation(self):
        return self._fugitive_observation

    def get_ground_truth_observation(self):
        return self._ground_truth_observation

    def get_blue_observation(self):
        return self._blue_observation

    def get_last_k_fugitive_detections(self):
        return self.last_k_fugitive_detections

    @property
    def cached_terrain_image(self):
        """
        cache terrain image to be more efficient when rendering
        :return:
        """
        # return self._cached_terrain_image
        return self.terrain.current_map_image

    def fast_render_canvas(self, show=True, scale=3, predicted_prisoner_location=None, show_delta=False):
        """
        We allow the predicted prisoner location to be passed in which renders a predicted prisoner location
        show_delta: is a bool whether or not to display the square around the fugitive
        """
        # Init the canvas
        self.canvas = self.cached_terrain_image
        self.canvas = cv2.flip(self.canvas, 0)

        def calculate_appropriate_image_extent_cv(loc, radius=0.4):
            y_new = -loc[1] + self.y_dim_render
            return list(map(int, [max(loc[0] - radius, 0), min(loc[0] + radius, self.x_dim_render),
                                  max(y_new - radius, 0), min(y_new + radius, self.y_dim_render)]))

        def draw_radius_of_detection(location, radius):
            radius = int(radius * self.ratio_render)
            color = (0, 0, 1)  # red detection circle
            location = (int(location[0] * self.x_render_scale), self.y_dim_render - int(location[1] * self.y_render_scale))
            cv2.circle(self.canvas, location, radius, color, 2)

        def draw_image_on_canvas_cv(image, location, asset_size):
            asset_size = int(asset_size)
            if asset_size % 2 != 0:
                asset_size = asset_size - 1
            if asset_size == 0:
                asset_size = 2

            location = (int(location[0] * self.x_render_scale), int(location[1] * self.y_render_scale))

            x_min, x_max, y_min, y_max = calculate_appropriate_image_extent_cv(location, asset_size)

            img = cv2.resize(image, (x_max - x_min, y_max - y_min))

            # create mask based on alpha channel
            mask = img[:, :, 3]
            mask[mask > 50] = 255
            mask = cv2.bitwise_not(mask)

            # cut out portion of the background where we want to paste image
            cut_background = self.canvas[y_min:y_max, x_min:x_max, :]
            img_with_background = cv2.bitwise_and(cut_background, cut_background, mask=mask) + img[:, :, 0:3] / 255

            # insert new image into background/canvas
            self.canvas[y_min:y_max, x_min:x_max, :] = img_with_background

        # fugitive_speed = prisoner.
        if self.is_detected:
            draw_image_on_canvas_cv(self.detected_prisoner_pic_cv, self.prisoner.location, self.default_asset_size)
        else:
            draw_image_on_canvas_cv(self.prisoner_pic_cv, self.prisoner.location, self.default_asset_size)
        draw_radius_of_detection(self.prisoner.location, self.prisoner.detection_range)

        # draw predicted prisoner location
        if predicted_prisoner_location is not None:
            # flip for canvas
            predicted_prisoner_location[1] = self.dim_y - predicted_prisoner_location[1]
            cv2.circle(self.canvas, predicted_prisoner_location, 20, (0, 0, 1), -1)

        # towns
        for town in self.town_list:
            draw_image_on_canvas_cv(self.town_pic_cv, town.location, self.default_asset_size)
        # search parties
        for search_party in self.search_parties_list:
            draw_image_on_canvas_cv(self.search_party_pic_cv, search_party.location, self.default_asset_size)
            draw_radius_of_detection(search_party.location,
                                     search_party.base_100_pod_distance(self.current_prisoner_speed))
            draw_radius_of_detection(search_party.location,
                                     search_party.base_100_pod_distance(self.current_prisoner_speed) * 3)

        # helicopters
        if self.is_helicopter_operating():
            for helicopter in self.helicopters_list:
                draw_image_on_canvas_cv(self.helicopter_pic_cv, helicopter.location, self.default_asset_size)
                draw_radius_of_detection(helicopter.location,
                                         helicopter.base_100_pod_distance(self.current_prisoner_speed))
                draw_radius_of_detection(helicopter.location,
                                         helicopter.base_100_pod_distance(self.current_prisoner_speed) * 3)
        else:
            for helicopter in self.helicopters_list:
                draw_image_on_canvas_cv(self.helicopter_no_pic_cv, helicopter.location, self.default_asset_size)

        if show_delta:
            # Added by Manisha (Check first before pushing changes) delta = 0.05 = 121.4 on the map
            x1, y1 = self.prisoner.location[0] - 121, 2428 - self.prisoner.location[1] + 121
            x2, y2 = self.prisoner.location[0] + 121, 2428 - self.prisoner.location[1] - 121
            cv2.rectangle(self.canvas, (x1, y1), (x2, y2), (0, 0, 1), 2)

        # hideouts
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                draw_image_on_canvas_cv(self.known_hideout_pic_cv, hideout.location, self.hideout_radius * self.ratio_render * 3)
            else:
                draw_image_on_canvas_cv(self.unknown_hideout_pic_cv, hideout.location, self.hideout_radius * self.ratio_render * 3)

        for rendezvous in self.rendezvous_point_list:
            if rendezvous.reached:
                draw_image_on_canvas_cv(self.rendezvous_pic_check_cv, rendezvous.location, self.default_asset_size)
            else:
                draw_image_on_canvas_cv(self.rendezvous_pic_cv, rendezvous.location, self.default_asset_size)

        # cameras
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                draw_image_on_canvas_cv(self.known_camera_pic_cv, camera.location, camera.detection_range* self.ratio_render)
            else:
                draw_image_on_canvas_cv(self.unknown_camera_pic_cv, camera.location, camera.detection_range* self.ratio_render)

        if show:
            cv2.imshow("test", self.canvas)
            cv2.waitKey(1)
        return (self.canvas * 255).astype('uint8')

    def render(self, mode, show=True, fast=False, scale=3, show_delta=False):
        """
        Render the environment.
        :param mode: required by `gym.Env` but we ignore it
        :param show: whether to show the rendered image
        :param fast: whether to use the fast version for render. The fast version takes less time to render but the render quality is lower.
        :param scale: scale for fast render
        :param show_delta: is a bool whether or not to display the square around the fugitive
        :return: opencv img object
        """
        if fast:
            return self.fast_render_canvas(show, scale, show_delta=show_delta)
        else:
            return self.slow_render_canvas(show)

    def slow_render_canvas(self, show=True):
        """
        Provide a visualization of the current status of the environment.

        In rendering, imshow interprets the matrix as:
        [x, 0]
        ^
        |
        |
        |
        |
        |----------->[0, y]
        However, the extent of the figure is still:
        [0, y]
        ^
        |
        |
        |
        |
        |----------->[x, 0]
        Read https://matplotlib.org/stable/tutorials/intermediate/imshow_extent.html for more explanations.

        :param show: whether to show the visualization directly or just return
        :return: an opencv img object
        """

        def calculate_appropriate_image_extent(loc, radius=0.4):
            """
            :param loc: the center location to put a picture
            :param radius: the radius (size) of the figure
            :return: [left, right, bottom, top]
            """
            return [max(loc[0] - radius, 0), min(loc[0] + radius, self.dim_x),
                    max(loc[1] - radius, 0), min(loc[1] + radius, self.dim_y)]

        fig, ax = plt.subplots(figsize=(20, 20))
        # Show terrain
        im = ax.imshow(self.cached_terrain_image, origin='lower')
        # labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # prisoner_history
        prisoner_location_history = np.array(self.prisoner_location_history)
        ax.plot(prisoner_location_history[:, 0], prisoner_location_history[:, 1], "r")

        # prisoner
        if self.is_detected:
            ax.imshow(self.detected_prisoner_pic,
                      extent=calculate_appropriate_image_extent(self.prisoner.location, radius=50))
        else:
            ax.imshow(self.prisoner_pic, extent=calculate_appropriate_image_extent(self.prisoner.location, radius=50))

        # towns
        for town in self.town_list:
            ax.imshow(self.town_pic, extent=calculate_appropriate_image_extent(town.location, radius=30))
        # search parties
        for search_party in self.search_parties_list:
            ax.imshow(self.search_party_pic, extent=calculate_appropriate_image_extent(search_party.location,
                                                                                       radius=search_party.detection_range))
        # helicopters
        if self.is_helicopter_operating():
            for helicopter in self.helicopters_list:
                ax.imshow(self.helicopter_pic, extent=calculate_appropriate_image_extent(helicopter.location,
                                                                                         radius=helicopter.detection_range))
        # hideouts
        for hideout in self.hideout_list:
            if hideout.known_to_good_guys:
                ax.imshow(self.known_hideout_pic,
                          extent=calculate_appropriate_image_extent(hideout.location, radius=self.hideout_radius))
            else:
                ax.imshow(self.unknown_hideout_pic,
                          extent=calculate_appropriate_image_extent(hideout.location, radius=self.hideout_radius))

        # cameras
        for camera in self.camera_list:
            if camera.known_to_fugitive:
                ax.imshow(self.known_camera_pic, extent=calculate_appropriate_image_extent(camera.location,
                                                                                           radius=camera.detection_range))
            else:
                ax.imshow(self.unknown_camera_pic, extent=calculate_appropriate_image_extent(camera.location,
                                                                                             radius=camera.detection_range))
        # finalize
        ax.axis('scaled')
        plt.savefig("simulator/temp.png")
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if show:
            plt.show()
        plt.close()

        return img

    def get_prisoner_location(self):
        return self.prisoner.location

    def generate_policy_heatmap(self, current_state, policy, num_timesteps=2500, num_rollouts=20, end=False):
        """
        Generates the heatmap displaying probabilities of ending up in certain cells
        :param current_state: current location of prisoner, current state of world
        :param policy: must input state, output action
        :param num_timesteps: how far in time ahead, remember time is in 15 minute intervals.
        """

        # Create 2D matrix
        display_matrix = np.zeros((self.dim_x + 1, self.dim_y + 1))

        for num_traj in tqdm(range(num_rollouts), desc="generating_heatmap"):
            observation = self.reset()
            for j in range(num_timesteps):
                if policy == 'rand':
                    action = self.action_space.sample()
                else:
                    action = policy.predict(observation, deterministic=False)[0]
                    # action = policy(observation)
                    # theta = policy([observation])
                    # action = np.array([7.5, theta[0]], dtype=np.float32)
                observation, reward, done, _ = self.step(action)
                # update count
                if not end:
                    display_matrix[self.prisoner.location[0], self.dim_y - self.prisoner.location[1]] += 4
                if done:
                    if end:
                        display_matrix[self.prisoner.location[0], self.dim_y - self.prisoner.location[1]] += 4
                    break
            if end:
                display_matrix[self.prisoner.location[0], self.dim_y - self.prisoner.location[1]] += 4
                # self.render('human', show=True)
        fig, ax = plt.subplots()
        display_matrix = np.transpose(display_matrix)
        from scipy.ndimage import gaussian_filter
        # smooth the matrix
        smoothed_matrix = gaussian_filter(display_matrix, sigma=50)
        # Set 0s to None as they will be ignored when plotting
        # smoothed_matrix[smoothed_matrix == 0] = None
        display_matrix[display_matrix == 0] = None
        # Plot the data
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                sharex=False, sharey=True,
                                figsize=(5, 5))
        # ax1.matshow(display_matrix, cmap='hot')
        # ax1.set_title("Original matrix")
        im = ax1.matshow(smoothed_matrix)
        num_hours = str((num_timesteps / 60).__round__(2))

        ax1.set_title("Heatmap at Time t=" + num_hours + ' hours')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_ticks([])
        plt.tight_layout()
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        cbar.ax.invert_yaxis()
        plt.show()

        print("saving heatmap")
        plt.savefig("simulator/temp.png")


if __name__ == "__main__":
    np.random.seed(20)