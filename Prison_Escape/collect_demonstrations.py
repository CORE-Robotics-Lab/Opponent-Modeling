""" This script collects a triple of agent observations, hideout location, and timestep. """
import os, sys

sys.path.append(os.getcwd())

import numpy as np
from tqdm import tqdm
from Prison_Escape.environment.gnn_wrapper import PrisonerGNNEnv

from Prison_Escape.environment.load_environment import load_environment
from Prison_Escape.environment.prisoner_perspective_envs import PrisonerEnv
from Prison_Escape.environment import PrisonerBothEnv, PrisonerBlueEnv, PrisonerEnv
from blue_policies.heuristic import BlueHeuristic
from fugitive_policies.heuristic import HeuristicPolicy
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
from fugitive_policies.a_star_avoid import AStarAdversarialAvoid

from Prison_Escape.environment import initialize_prisoner_environment
import argparse
import random


def collect_demonstrations(epsilon, num_runs,
                           starting_seed,
                           random_cameras,
                           folder_name,
                           heuristic_type,
                           env_path,
                           show=False):
    """ Collect demonstrations for the homogeneous gnn where we assume all agents are the same.
    :param env: Environment to collect demonstrations from
    :param policy: Policy to use for demonstration collection
    :param repeat_stops: Number of times to repeat the last demonstration

    """
    num_detections = 0
    total_timesteps = 0
    detect_rates = []
    for seed in tqdm(range(starting_seed, starting_seed + num_runs)):
        detect = 0
        t = 0
        agent_observations = []
        hideout_observations = []
        timestep_observations = []
        detected_locations = []
        blue_observations = []
        red_observations = []
        last_k_fugitive_detections = []

        red_locations = []
        dones = []

        num_random_known_cameras = random.randint(25, 30)
        num_random_unknown_cameras = random.randint(25, 30)

        env = load_environment(env_path)
        env.set_seed(seed)
        print("Running with seed {}".format(seed))
        np.random.seed(seed)
        random.seed(seed)

        if heuristic_type == 'Normal':
            red_policy = HeuristicPolicy(env, epsilon=epsilon)
            # path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_eps_{epsilon}_{heuristic_type}"
            path = f"/data/manisha/datasets/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
        elif heuristic_type == 'AStar':
            red_policy = AStarAdversarialAvoid(env, cost_coeff=1000)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
        else:
            red_policy = RRTStarAdversarialAvoid(env, max_speed=7.5, n_iter=2000)
            path = f"datasets/{folder_name}/gnn_map_{map_num}_run_{num_runs}_{heuristic_type}"
        if random_cameras:
            path += "_random_cameras"

        env = PrisonerBlueEnv(env, red_policy)
        if not os.path.exists(path):
            os.makedirs(path)

        env = PrisonerGNNEnv(env)
        policy = BlueHeuristic(env, debug=False)

        gnn_obs, blue_obs = env.reset()
        policy.reset()
        policy.init_behavior()
        got_stuck = False
        done = False
        while not done:
            t += 1
            blue_actions = policy.predict(blue_obs)
            gnn_obs, blue_obs, reward, done, _ = env.step(blue_actions)

            prisoner_location = env.get_prisoner_location()
            # detected_location = blue_obs[-2:]

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
                if env.unwrapped.timesteps >= 2000:
                    print(f"Got stuck, {env.unwrapped.timesteps}")
                    got_stuck = True
                detect_rates.append(detect / t)
                print(f"{detect}/{t} = {detect / t} detection rate")
                break

        agent_dict = {"num_known_cameras": env.num_known_cameras,
                      "num_unknown_cameras": env.num_unknown_cameras,
                      "num_helicopters": env.num_helicopters,
                      "num_search_parties": env.num_search_parties}

        blue_obs_dict = env.blue_obs_names._idx_dict
        prediction_obs_dict = env.prediction_obs_names._idx_dict

        if not got_stuck:
            np.savez(path + f"/seed_{seed}_known_{env.num_known_cameras}_unknown_{env.num_unknown_cameras}.npz",
                     blue_observations=blue_observations,
                     red_observations=red_observations,
                     agent_observations=agent_observations,
                     hideout_observations=hideout_observations,
                     timestep_observations=timestep_observations,
                     detected_locations=detected_locations,
                     red_locations=red_locations,
                     dones=dones,
                     agent_dict=agent_dict,
                     detect=detect,
                     last_k_fugitive_detections=last_k_fugitive_detections,
                     blue_obs_dict=blue_obs_dict,
                     prediction_obs_dict=prediction_obs_dict
                     )
    print(detect_rates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_num', type=int, default=0, help='Environment to use')

    args = parser.parse_args()
    map_num = args.map_num

    store_last_k_fugitive_detections = True

    # starting_seed = 0; num_runs = 300; epsilon=0.1; folder_name = "train"
    # starting_seed = 500; num_runs = 100; epsilon = 0.1; folder_name = "test"
    starting_seed = 0  # Starting seed
    num_runs = 300  # Number of rollouts
    epsilon = 0.1
    folder_name = "train"  # train, val or test

    # Adjust the detection range in the config file below to obtain different datasets as described in the paper
    env_path = "Prison_Escape/environment/configs/balance_game.yaml"

    heuristic_type = "AStar"
    random_cameras = False
    observation_step_type = "Blue"

    # print(agent_observations.shape)
    collect_demonstrations(epsilon, num_runs, starting_seed, random_cameras, folder_name, heuristic_type, env_path,
                           show=True)