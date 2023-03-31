"""
This file is used as part of the input pipeline to the filtering network.
The file has functions to compute the sensor heatmap and fugitive likelihood based on the locations of the blue agents
and their Probability of Detecting (PoD) the fugitive.
The sensor heatmap and fugitive likelihood (based on the blue agent's obs) can later be used as prior information in the
Prediction-Filtering Network.
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from os.path import exists
from simulator.observation_spaces import create_observation_space_blue_team
from simulator.prisoner_env import PrisonerEnv
from simulator.forest_coverage.generate_square_map import generate_square_map

TINY = 1e-8


def softmax(x):
    """
    Returns softmax of x
    :param x: input
    :return: softmax(x)
    """
    return np.exp(x) / (np.sum(np.exp(x)) + TINY)


def get_heatmap_from_blue_data(batch_size, dim_x, dim_y, blue_agent_locations, blue_agent_detection_ranges):
    """
    Method to generate heatmap given locations and detection ranges for the blue agents
    :param batch_size: # Observations in the current batch
    :param dim_x: Terrain x-dimension.
    :param dim_y: Terrain y-dimension.
    :param blue_agent_locations: (np array) of locations of all the blue agents in each observation in the batch
    :param blue_agent_detection_ranges: (np array) of detection ranges of all the blue agents in each observation in the batch
    :return:
        sensor heatmap: (np array) Computed Sensor heatmap for each observation in the batch - Size(batch x dim_x x dim_y)
    """
    sensor_heatmap = np.zeros((batch_size, dim_x, dim_y))

    all_points = np.indices(sensor_heatmap.shape)
    # all_points[0] --> batch_indices, all_points[1] --> row_indices (x), all_points[2] --> column indices (y)

    total_blue_agents = blue_agent_locations.shape[1]

    for i in range(total_blue_agents):
        # First get coordinates in bounding box.
        # Then only for these coordinates, check if they lie within the PoD circle
        r = blue_agent_detection_ranges[:, i]  # detection range --> radius
        r_squared = (r ** 2).reshape(-1, 1)
        center_x, center_y = blue_agent_locations[:, i][:, 0].reshape(-1, 1), \
                             blue_agent_locations[:, i][:, 1].reshape(-1, 1)  # location --> center

        # Directly identify which points are within the detection range across all heatmaps in the batch

        # Broadcast values for comparison from batch x 1 to batch x dim_x x dim_y
        center_x = np.repeat(center_x, dim_x, axis=1)
        center_x = center_x[:, :, np.newaxis]
        center_x = np.repeat(center_x, dim_y, axis=2)

        center_y = np.repeat(center_y, dim_x, axis=1)
        center_y = center_y[:, :, np.newaxis]
        center_y = np.repeat(center_y, dim_y, axis=2)

        r_squared = np.repeat(r_squared, dim_x, axis=1)
        r_squared = r_squared[:, :, np.newaxis]
        r_squared = np.repeat(r_squared, dim_y, axis=2)

        # Calculate Distance between the points on grid and locations of the agents
        distance = (all_points[1] - center_x) ** 2 + (all_points[2] - center_y) ** 2

        # Check which points are within the range
        batch_idx = all_points[:, distance <= r_squared][0]
        x_idx = all_points[:, distance <= r_squared][1]
        y_idx = all_points[:, distance <= r_squared][2]

        # Modify values on the sensor heatmap based on PoD
        # If sensor ranges overlap, use 1 - (1-p1)*(1-p2) where p1 and p2 are the PoDs of the overlapping sensors
        PoD = (1 - distance / r_squared)
        sensor_heatmap[batch_idx, x_idx, y_idx] = (1 - (1 - sensor_heatmap) * (1 - PoD))[distance <= r_squared]

    return sensor_heatmap


def get_camera_heatmap(obs: np.array, idx_dict,
                       num_known_cameras=5, num_unknown_cameras=5, num_known_hideouts=1,
                       dim_x=2428, dim_y=2428, save_file=False):
    """
    Cameras are fixed in the env, and hence their heatmaps need not be computed for each observation in
    the mini-batch. Here, we extract the position and detection ranges of cameras, and stack it . This can be
    used to merge with the heatmap of other dynamic blue agents
    :param obs: (np array: m x k) batch of current blue observations (batch size=m) from the Prisoner Env (obs dim = k)
    :param idx_dict: dict with keys as names of agent_loc, agent_detection, and values as corresponding index values in
                    the obs. Received from wrap in Observation_Spaces.py
    :param num_known_cameras: # Cameras known to the fugitive in the terrain
    :param num_unknown_cameras: # Cameras unknown to the fugitive in the terrain
     :param num_known_hideouts: # Hideouts known to the blue agent. There's a camera placed at every known hideout
    :param dim_x: Terrain x-dimension. Default=2428
    :param dim_y: Terrain y-dimension. Default=2428
    :param save_file: (bool) If true, it saves the computed camera_heatmap numpy array
    :return:
        camera_heatmap: (np.array) Size (dim_x x dim_y). Pre-compute the camera_heatmap to be used in generating sensor
                        heatmaps for every batch.
    """

    # TODO: Store this as a pickle and just reload the numpy array during training??
    camera_heatmap = np.zeros((dim_x, dim_y))

    all_points = np.indices(camera_heatmap.shape)

    total_num_cameras = num_known_hideouts + num_unknown_cameras + num_known_cameras

    camera_detection_range = 30 * 2  # Assumption: Fixed detection range for all cameras
    r_squared = camera_detection_range ** 2  # Distance from the camera center squared (for computing range values)

    for camera_idx in range(total_num_cameras):
        location = obs[:, idx_dict['camera_loc_{}'.format(camera_idx)]] * 2428
        center_x, center_y = location[0, 0].reshape(-1, 1), location[0, 1].reshape(-1, 1)

        # Calculate Distance between the points on grid and locations of the agents
        distance = (all_points[0] - center_x) ** 2 + (all_points[1] - center_y) ** 2

        # Check which points are within the range of detection
        in_range_points = all_points[:, distance <= r_squared]
        x_idx = in_range_points[0]
        y_idx = in_range_points[1]

        # Modify values on the heatmap based on PoD
        # If sensor ranges overlap, use 1 - (1-p1)*(1-p2) where p1 and p2 are the PoDs of the overlapping sensors
        PoD = (1 - distance / r_squared)
        camera_heatmap[x_idx, y_idx] = (1 - (1 - camera_heatmap) * (1 - PoD))[distance <= r_squared]
    if save_file:
        with open('utils/camera_heatmap.npy', 'wb')as f:
            np.save(f, camera_heatmap)

    return camera_heatmap


def get_input_maps_from_blue_obs_batch(obs: np.array, num_known_cameras=5, num_unknown_cameras=5, num_helicopters=1,
                                       num_search_parties=2, num_known_hideouts=1, camera_heatmap=None):
    """
    Computes the sensor heatmap and fugitive likelihood maps from parameterized obs vector

    :param obs: (np array: m x k) batch of current blue observations (batch size=m) from the Prisoner Env (obs dim = k)
    :param num_known_cameras: # Cameras known to the fugitive in the terrain
    :param num_unknown_cameras: # Cameras unknown to the fugitive in the terrain
    :param num_helicopters: # Helicopter Agents
    :param num_search_parties: # Search Party Agents
    :param num_known_hideouts: # Hideouts known to the blue agent
    :param camera_heatmap: Pre-computed camera heatmap (as the camera locations are fixed, we don't need to compute this
                            everytime.

    :return:
        sensor_heatmap, fugitive_likelihood of the same dimension as the env
    """

    dim_x, dim_y = 2428, 2428
    obs_features = obs.shape[-1]

    # Reshape observations as batch x obs_features
    obs = obs.reshape(-1, obs_features)
    batch_size = obs.shape[0]

    fugitive_likelihood = np.zeros((batch_size, dim_x, dim_y))  # initialize grid
    sensor_heatmap = np.zeros((batch_size, dim_x, dim_y))  # initialize grid

    blue_obs_space, blue_obs_names = create_observation_space_blue_team(
        num_known_cameras=num_known_cameras + num_known_hideouts,
        num_unknown_cameras=num_unknown_cameras,
        num_helicopters=num_helicopters,
        num_search_parties=num_search_parties,
        num_known_hideouts=num_known_hideouts)

    assert blue_obs_space.shape[-1] == obs.shape[-1], "Observation shape Mismatch %s, %s. Cannot generate heatmap!" \
                                                      "Please check the number of blue agents in the env." \
                                                      % (blue_obs_space.shape[-1], obs.shape[-1])

    # List indices for extracting coordinates of blue agents' locations and detection ranges
    idx_names = blue_obs_names._names
    idx_dict = {}
    start = 0

    for name, l in idx_names:
        idx_dict[name] = np.arange(l) + start
        start += l

    total_blue_agents = num_helicopters + num_search_parties

    blue_agent_locations = np.zeros((batch_size, total_blue_agents, 2))
    blue_agent_detection_ranges = np.zeros((batch_size, total_blue_agents, 1))

    agent_idx = 0

    # Extract locations and detection ranges for search parties and helicopters
    for helo_idx in range(num_helicopters):
        location = obs[:, idx_dict['helicopter_location_{}'.format(helo_idx)]] * 2428
        detection_range = 90  # TODO: Fix how to get detection range of blue agents from obs
        blue_agent_locations[:, agent_idx, :] = location
        blue_agent_detection_ranges[:, agent_idx, :] = detection_range
        agent_idx += 1

    for search_party_idx in range(num_search_parties):
        location = obs[:, idx_dict['search_party_location_{}'.format(search_party_idx)]] * 2428
        detection_range = 135  # TODO: Fix how to get detection range of blue agents from obs
        blue_agent_locations[:, agent_idx, :] = location
        blue_agent_detection_ranges[:, agent_idx, :] = detection_range
        agent_idx += 1

    sensor_heatmap = get_heatmap_from_blue_data(batch_size, dim_x, dim_y, blue_agent_locations,
                                                blue_agent_detection_ranges)

    # Get Camera Heatmap for the batch
    if camera_heatmap is None:
        camera_heatmap = get_camera_heatmap(obs, idx_dict, num_known_cameras, num_unknown_cameras, num_known_hideouts,
                                            dim_x, dim_y, save_file=True)

    # Stack camera heatmaps to merge with ranges of other sensors
    camera_heatmap = np.stack([camera_heatmap] * batch_size, axis=0)
    assert camera_heatmap.shape == sensor_heatmap.shape, "Sensor heatmap does not match the dimensions of camera heatmap." \
                                                         "Please regenerate the camera heatmap"
    sensor_heatmap = 1 - (1 - sensor_heatmap) * (1 - camera_heatmap)

    percent_dense = 0.30
    size_of_dense_forest = int(dim_x * percent_dense)
    forest_coverage = generate_square_map(size_of_dense_forest=size_of_dense_forest, dim_x=dim_x, dim_y=dim_y)
    sensor_heatmap = sensor_heatmap * forest_coverage  # Modify detection values based on tl coverage

    # Prisoner Location is the last two indices of the blue observation
    # prisoner_detected = np.zeros((batch_size,))
    prisoner_locations = obs[:, -2:]
    prisoner_detected = 1 - (prisoner_locations[:, 0] == -1)
    detected_locations = (prisoner_locations[prisoner_detected == 1] * dim_x).astype(int)

    # If prisoner was detected by at least one blue agent
    fugitive_likelihood[prisoner_detected == 1, detected_locations[:, 0], detected_locations[:, 1]] = 1

    # If no blue agent saw the prisoner
    fugitive_likelihood[prisoner_detected == 0] = softmax(1 - sensor_heatmap)[prisoner_detected == 0]

    return sensor_heatmap, fugitive_likelihood


if __name__ == '__main__':
    env = PrisonerEnv(spawn_mode='uniform', observation_step_type="Blue", random_cameras=False,
                      camera_file_path="simulator/camera_locations/fill_camera_locations.txt",
                      place_mountains_bool=False, camera_range_factor=2.0)

    # env = PrisonerEnv(observation_step_type="Blue", random_cameras=False)  # old env with few cameras

    # start_time = time.time()

    # Get attributes from the env
    num_known_hideouts = env.num_known_hideouts
    num_known_cameras = env.num_known_cameras - num_known_hideouts
    num_unknown_cameras = env.num_unknown_cameras
    num_search_parties = env.num_search_parties
    num_helicopters = env.num_helicopters

    # obs = env.reset().reshape(1, -1)
    obs = np.stack([env.reset(), env.reset(), env.reset()])

    # Pre-load camera heatmap if file exists
    camera_heatmap = None
    if exists('utils/camera_heatmap.npy'):
        camera_heatmap = np.load('utils/camera_heatmap.npy')

    sensor_heatmap, fugitive_likelihood = get_input_maps_from_blue_obs_batch(obs=obs,
                                                                             num_known_cameras=num_known_cameras,
                                                                             num_unknown_cameras=num_unknown_cameras,
                                                                             num_helicopters=num_helicopters,
                                                                             num_search_parties=num_search_parties,
                                                                             num_known_hideouts=num_known_hideouts,
                                                                             camera_heatmap=camera_heatmap)

    # print("Execution time: {}".format(time.time() - start_time))
