import numpy as np
from gym import spaces
# from simulator.prisoner_env import ObservationNames

class ObservationNames:
    """Helper class to reference an observation's elements by name"""
    class NamedObservation:
        """Wrapper for an ndarray to allow its elements to be accessed according to a specific naming scheme. Instantiated by an ObservationWrapper instance."""
        def __init__(self, array, names):
            self.names = names
            self.array = array
        def __getitem__(self, key):
            s, t = self.names[key]
            return self.array[s:t]
        def __setitem__(self, key, value):
            s, t = self.names[key]
            self.array[s:t] = value
        def __repr__(self):
            return repr({name: self[name] for name in self.names})
    def __init__(self):
        self._names = []
        self._idx_dict = {}
    def add_name(self, name, length):
        self._names.append((name, length))
        self._idx_dict = {}
        k = 0
        for name, l in self._names:
            self._idx_dict[name] = k, k+l
            k += l
    def wrap(self, array):
        return ObservationNames.NamedObservation(array, self._idx_dict)
    def __call__(self, array):
        return self.wrap(array)

def create_observation_space_ground_truth(num_known_cameras, num_unknown_cameras, num_known_hideouts, num_unknown_hideouts, num_helicopters, num_search_parties, terrain_size=0):
        """ Create observation space for ground truth
        :param num_known_cameras: number of known cameras
        :param num_unknown_cameras: number of unknown cameras
        :param num_known_hideouts: number of known hideouts
        :param num_unknown_hideouts: number of unknown hideouts
        :param num_helicopters: number of helicopters
        :param num_search_parties: number of search parties
        :param terrain_size: size of the terrain feature vector, currently using autoencoder to generate compressed feature space
        
        """
        
        # observation and action spaces (for fugitive)
        obs_names = ObservationNames()
        # observation space contains
        # 1. time (divided by 4320 to normalize)
        observation_low = [0]
        observation_high = [1]
        obs_names.add_name('time', 1)
        # 2.1 location of all cameras (divided by 2428 to normalize)
        for i in range(num_known_cameras + num_unknown_cameras):
            observation_low.extend([0, 0])
            observation_high.extend([1, 1])
            obs_names.add_name('camera_loc_%d'%i, 2)
        # 2.2 [b, x,y] are locations of hideouts (divided by 2428 to normalize)
        for i in range(num_known_hideouts + num_unknown_hideouts):
            observation_low.extend([0, 0, 0])
            observation_high.extend([1, 1, 1])
            obs_names.add_name('hideout_known_%d'%i, 1)
            obs_names.add_name('hideout_loc_%d'%i, 2)
        # 3. prisoner location (divided by 2428), self speed (divided by 15), self heading (divided by pi)
        observation_low.extend([0, 0, 1.0 / 15, -1])
        observation_high.extend([1, 1, 1, 1])
        obs_names.add_name('prisoner_loc', 2)
        obs_names.add_name('prev_action', 2)
        # 4. helicopter and search party locations, we're not including detection here
        for i in range(num_helicopters):
            observation_low.extend([0, 0,])
            observation_high.extend([1, 1])
            obs_names.add_name('helicopter_location_%d'%i, 3)
        for i in range(num_search_parties):
            observation_low.extend([0, 0,])
            observation_high.extend([1, 1])
            obs_names.add_name('search_party_location_%d'%i, 3)

        # Fugitive detection of parties
        for i in range(num_helicopters):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('helicopter_detect_%d'%i, 3)
        for i in range(num_search_parties):
            observation_low.extend([0])
            observation_high.extend([1])

        # Party detection of fugitive
        for _ in range(num_known_cameras):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('known_camera_%d'%i, 3)
        for _ in range(num_unknown_cameras):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('unknown_camera_%d'%i, 3)
            # parties_detection_of_fugitive.extend(camera.detect(prisoner.location, speed))
        for _ in range(num_helicopters):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('helicopter_%d'%i, 3)
            # parties_detection_of_fugitive.extend(helicopter.detect(prisoner.location, speed))
        for _ in range(num_search_parties):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('search_party_%d'%i, 3)

        # Terrain shape
        observation_low.extend([0]*terrain_size)
        observation_high.extend([1]*terrain_size)

        observation_high = np.array(observation_high, dtype=np.float)
        observation_low = np.array(observation_low, dtype=np.float)

        observation_space = spaces.Box(observation_low, observation_high)
        return observation_space, obs_names

def create_observation_space_fugitive(num_known_cameras, num_known_hideouts, num_unknown_hideouts, num_helicopters, num_search_parties, num_rendezvous_points, terrain_size=0):
    """ Create observation space for fugitive 
    :param num_known_cameras: number of known cameras
    :param num_known_hideouts: number of known hideouts
    :param num_unknown_hideouts: number of unknown hideouts
    :param num_helicopters: number of helicopters
    :param num_search_parties: number of search parties
    :param terrain_size: size of the terrain feature vector, currently using autoencoder to generate compressed feature space
    
    """
    
    # observation and action spaces (for fugitive)
    obs_names = ObservationNames()
    # observation space contains
    # 1. time (divided by 4320 to normalize)
    observation_low = [0]
    observation_high = [1]
    obs_names.add_name('time', 1)
    # 2.1 location of known cameras (divided by 2428 to normalize)
    for i in range(num_known_cameras):
        observation_low.extend([0, 0])
        observation_high.extend([1, 1])
        obs_names.add_name('camera_loc_%d'%i, 2)
    # 2.2 [b,x,y] where b is whether the hideout is known to the good guys and [x,y] are locations of hideouts (divided by 2428 to normalize)
    for i in range(num_known_hideouts+num_unknown_hideouts):
        observation_low.extend([0, 0, 0])
        observation_high.extend([1, 1, 1])
        obs_names.add_name('hideout_known_%d'%i, 1)
        obs_names.add_name('hideout_loc_%d'%i, 2)
    # 3. self location (divided by 2428), self speed (divided by 15), self heading (divided by pi)
    observation_low.extend([0, 0, 1.0 / 15, -1])
    observation_high.extend([1, 1, 1, 1])
    obs_names.add_name('prisoner_loc', 2)
    obs_names.add_name('prev_action', 2)
    # 4. detection of [helicopters, helicopter dropped cameras (currently not implemented), search parties]
    # Detection is encoded by a three tuple [b, x, y] where b in binary. If b=1 (detected), [x, y] will have the detected location in world coordinates. If b=0 (not detected), [x, y] will be [-1, -1].
    for i in range(num_helicopters):
        observation_low.extend([0, -1, -1])
        observation_high.extend([1, 1, 1])
        obs_names.add_name('helicopter_detect_%d'%i, 3)
    for i in range(num_search_parties):
        observation_low.extend([0, -1, -1])
        observation_high.extend([1, 1, 1])
        obs_names.add_name('search_party_detect_%d'%i, 3)

    # Add rendezvous points on the map
    # for i in range(num_rendezvous_points):
    observation_low.extend([0, 0])
    observation_high.extend([1, 1])
    obs_names.add_name('rendezvous_point', 2)

    # Terrain shape
    observation_low.extend([0]*terrain_size)
    observation_high.extend([1]*terrain_size)

    observation_high = np.array(observation_high, dtype=np.float)
    observation_low = np.array(observation_low, dtype=np.float)

    observation_space = spaces.Box(observation_low, observation_high)
    return observation_space, obs_names

def create_observation_space_prediction(num_known_cameras, num_known_hideouts, num_helicopters, num_search_parties, terrain_size):
    # observation and action spaces (for fugitive)
    obs_names = ObservationNames()
    # observation space contains
    # 1. time (divided by 4320 to normalize)
    observation_low = [0]
    observation_high = [1]
    obs_names.add_name('time', 1)
    # 2.1 location of known cameras (divided by 2428 to normalize)
    for i in range(num_known_cameras):
        observation_low.extend([0, 0])
        observation_high.extend([1, 1])
        obs_names.add_name('camera_loc_%d'%i, 2)
    # 2.2 [x,y] where [x,y] are locations of hideouts (divided by 2428 to normalize). All hideouts are in this observation are known hideouts
    for i in range(num_known_hideouts):
        observation_low.extend([0, 0])
        observation_high.extend([1, 1])
        obs_names.add_name('hideout_loc_%d'%i, 2)
    # 3. self location (divided by 2428), self speed (divided by 15), self heading (divided by pi)
    observation_low.extend([0, 0, 1.0 / 15, -1])
    observation_high.extend([1, 1, 1, 1])
    obs_names.add_name('prisoner_loc', 2)
    obs_names.add_name('prev_action', 2)
    # 4. detection of [helicopters, helicopter dropped cameras (currently not implemented), search parties]
    # Detection is encoded by a three tuple [b, x, y] where b in binary. If b=1 (detected), [x, y] will have the detected location in world coordinates. If b=0 (not detected), [x, y] will be [-1, -1].
    for i in range(num_helicopters):
        observation_low.extend([0, -1, -1])
        observation_high.extend([1, 1, 1])
        obs_names.add_name('helicopter_detect_%d'%i, 3)
    for i in range(num_search_parties):
        observation_low.extend([0, -1, -1])
        observation_high.extend([1, 1, 1])
        obs_names.add_name('search_party_detect_%d'%i, 3)

    # Terrain shape
    observation_low.extend([0]*terrain_size)
    observation_high.extend([1]*terrain_size)

    observation_high = np.array(observation_high, dtype=np.float)
    observation_low = np.array(observation_low, dtype=np.float)

    observation_space = spaces.Box(observation_low, observation_high)
    return observation_space, obs_names

def create_observation_space_blue_team(num_known_cameras, num_unknown_cameras, num_known_hideouts, num_helicopters, num_search_parties, terrain_size=0, include_start_location_blue_obs=False):
        """ Create observation space for blue team 
        :param num_known_cameras: number of known cameras
        :param num_unknown_cameras: number of unknown cameras
        :param num_known_hideouts: number of known hideouts
        :param num_helicopters: number of helicopters
        :param num_search_parties: number of search parties
        :param terrain_size: size of the terrain feature vector, currently using autoencoder to generate compressed feature space
        
        """
        obs_names = ObservationNames()

        # 1. time (divided by 4320 to normalize)
        observation_low = [0]
        observation_high = [1]
        obs_names.add_name('time', 1)
        # 2.1 location of all cameras (divided by 2428 to normalize)
        for i in range(num_known_cameras):
            observation_low.extend([0, 0])
            observation_high.extend([1, 1])
            obs_names.add_name('known_camera_loc_%d'%i, 2)

        for i in range(num_unknown_cameras):
            observation_low.extend([0, 0])
            observation_high.extend([1, 1])
            obs_names.add_name('unknown_camera_loc_%d'%i, 2)
        # 2.2 [x,y] are locations of known hideouts (divided by 2428 to normalize)
        for i in range(num_known_hideouts):
            observation_low.extend([0, 0])
            observation_high.extend([1, 1])
            obs_names.add_name('hideout_loc_%d'%i, 2)
        # 3. helicopter and search party locations, we're not including detection here
        for i in range(num_helicopters):
            observation_low.extend([0, 0])
            observation_high.extend([1, 1])
            obs_names.add_name('helicopter_location_%d'%i, 2)
        for i in range(num_search_parties):
            observation_low.extend([0, 0])
            observation_high.extend([1, 1])
            obs_names.add_name('search_party_location_%d'%i, 2)

        #4. detection of fugitive by parties, appended here as this is how observation is constructed
        for i in range(num_known_cameras):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('known_camera_%d'%i, 1)
        for i in range(num_unknown_cameras):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('unknown_camera_%d'%i, 1)
        for i in range(num_helicopters):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('helicopter_%d'%i, 1)
        for i in range(num_search_parties):
            observation_low.extend([0])
            observation_high.extend([1])
            obs_names.add_name('search_party_%d'%i, 1)
        
        #5. If the prisoner was detected by ANY of the detection objects return location, else, [-1, -1]
        observation_low.extend([-1, -1])
        observation_high.extend([1, 1])
        obs_names.add_name('prisoner_detected', 2)

        # Terrain shape
        observation_low.extend([0]*terrain_size)
        observation_high.extend([1]*terrain_size)

        # include start location of fugitive into observation
        if include_start_location_blue_obs:
            observation_low.extend([0, 0])
            observation_high.extend([1, 1])
            
        observation_high = np.array(observation_high, dtype=np.float)
        observation_low = np.array(observation_low, dtype=np.float)
        observation_space = spaces.Box(observation_low, observation_high)
        return observation_space, obs_names

def transform_blue_detection_of_fugitive(parties_detection_of_fugitive):
    """ This is used to remove the repeated detections of the fugitive in the blue parties observation space.
    
        args: parties_detection_of_fugitive is a list of [b, x, y] where b is binary and [x, y] is the location of the detected fugitive for each agent
        returns: a list showing just if the prisoner was detected for each agent and the location of the agent if detected
    """
    one_hot = parties_detection_of_fugitive[::3]
    if not any(one_hot):
        return one_hot + [-1, -1]
    else:
        index = one_hot.index(1) # locate which agent detected the fugitive
        return one_hot + [i for i in parties_detection_of_fugitive[index*3+1:index*3+3]]
