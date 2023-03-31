""" This script collects a triple of agent observations, hideout location, and timestep. """
import os, sys
sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
from simulator.gnn_wrapper import SmugglerGNNEnv

from simulator.load_environment import load_environment
from simulator.smuggler_perspective_envs import SmugglerEnv, SmugglerBlueEnv
from simulator.smuggler_env import SmugglerBothEnv
# from simulator import SmugglerBothEnv, SmugglerBlueEnv, SmugglerEnv
from blue_policies.heuristic import BlueHeuristic
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid
from fugitive_policies.a_star_policy import AStarPolicy

# from simulator import initialize_prisoner_environment
import argparse
import random
import matplotlib.pyplot as plt

def collect_demonstrations(epsilon, num_runs, 
                    starting_seed, 
                    folder_name,
                    heuristic_type,
                    include_camera_at_start=False,
                    show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same. 
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration
    
    """
    env = SmugglerBothEnv(terrain=None,
                            spawn_mode='normal',
                            observation_step_type="Fugitive",
                            random_cameras=False,
                            camera_file_path="simulator/camera_locations/smuggler.txt",
                            camera_range_factor=1.0,
                            observation_terrain_feature=False,
                            random_hideout_locations=False,
                            spawn_range=350,
                            helicopter_battery_life=200,
                            helicopter_recharge_time=40,
                            num_search_parties=2,
                            num_helicopters=2,  # Change to 3 for Smuggler-High Dataset
                            camera_net_bool=False,
                            num_random_known_cameras=1,
                            num_random_unknown_cameras=1,
                            )
                            
    if heuristic_type == 'AStar':
        red_policy = AStarPolicy(env, cost_coeff=1)
        path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
    else:
        red_policy = RRTStarAdversarialAvoid(env, max_speed=7.5, n_iter=2000)
        path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"

    env = SmugglerBlueEnv(env, red_policy)
    if not os.path.exists(path):
        os.makedirs(path)

    env = SmugglerGNNEnv(env)
    policy = BlueHeuristic(env, debug=False)

    detection_rates = []
    reach_known = 0
    reach_unknown = 0
    for seed in tqdm(range(starting_seed, starting_seed + num_runs)):
        print("Running with seed {}".format(seed))
        detect = 0
        t = 0
        agent_observations = []
        hideout_observations = []
        timestep_observations = []
        detected_locations = []
        blue_observations = []
        red_observations = []
        last_k_fugitive_detections = []

        num_detections = 0
        total_timesteps = 0
        

        red_locations = []
        dones = []

        gnn_obs, blue_obs = env.reset(seed=seed)
        policy.reset()
        policy.init_behavior()

        done = False
        while not done:
            t += 1
            blue_actions = policy.predict(blue_obs)
            gnn_obs, blue_obs, reward, done, _ = env.step(blue_actions)
            # print(reward)

            prisoner_location = env.get_prisoner_location()
            
            blue_obs_wrapped = env.blue_obs_names(blue_obs)
            detected_location = blue_obs_wrapped["prisoner_detected"]

            blue_observation = env.get_blue_observation()
            red_observation = env.get_prediction_observation()

            red_observations.append(red_observation)
            blue_observations.append(blue_observation)
            agent_observations.append(gnn_obs[0])
            hideout_observations.append(gnn_obs[1])
            timestep_observations.append(gnn_obs[2])
            red_locations.append(prisoner_location)
            dones.append(done)
            detected_locations.append(detected_location)

            if env.is_detected:
                num_detections += 1
                detect += 1

            if store_last_k_fugitive_detections:
                last_k_fugitive_detections.append(np.array(env.get_last_k_fugitive_detections()))

            if show:
                env.render('heuristic', show=True, fast=True)

            if done:
                if reward == -1:
                    reach_known += 1
                else:
                    reach_unknown += 1
                if env.unwrapped.timesteps >= 2000:
                    print(f"Got stuck, {env.unwrapped.timesteps}")
                print(f"{detect}/{t} = {detect/t} detection rate")
                break
                
        agent_dict = {"num_known_cameras": env.num_known_cameras,
                    "num_unknown_cameras": env.num_unknown_cameras,
                    "num_helicopters": env.num_helicopters,
                    "num_search_parties": env.num_search_parties}

        blue_obs_dict = env.blue_obs_names._idx_dict
        prediction_obs_dict = env.prediction_obs_names._idx_dict
        np.savez(path + f"/seed_{seed}_known_{env.num_known_cameras}_unknown_{env.num_unknown_cameras}.npz", 
            blue_observations = blue_observations,
            red_observations = red_observations,
            agent_observations=agent_observations,
            hideout_observations=hideout_observations,
            timestep_observations=timestep_observations, 
            detected_locations = detected_locations,
            red_locations=red_locations, 
            dones=dones,
            agent_dict = agent_dict,
            detect=detect,
            last_k_fugitive_detections=last_k_fugitive_detections,
            blue_obs_dict = blue_obs_dict,
            prediction_obs_dict = prediction_obs_dict
            )
        detection_rates.append(detect/t)

    # plt histogram
    plt.figure()
    plt.hist(detection_rates, bins=20)
    plt.xlabel('Detection rate per Episode')
    plt.ylabel('Frequency')
    plt.savefig('figures/detection_rate_histogram.png')

    plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    names = ['Reach Known', 'Reach Unknown']
    amounts = [reach_known, reach_unknown]
    plt.bar(names,amounts)
    plt.savefig('figures/reach_graph.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')
    
    args = parser.parse_args()
    map_num = args.map_num

    include_camera_at_start=False
    store_last_k_fugitive_detections = True
    starting_seed = 0; num_runs = 10; epsilon=0; folder_name = "smuggler_astar"

    if include_camera_at_start:
        folder_name += "_include_camera_at_start"

    heuristic_type = "AStar"
    observation_step_type = "Blue"

    collect_demonstrations(epsilon, num_runs, starting_seed, folder_name, heuristic_type, include_camera_at_start=include_camera_at_start, show=True)