from simulator import SmugglerEnv
from fugitive_policies.heuristic import HeuristicPolicy
import os
import numpy as np
import cv2
from datetime import datetime
import random
import sys

sys.path.append(os.getcwd())

def save_video(ims, filename, fps=30.0):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()


sys.path.append(os.getcwd())

os.makedirs("logs/temp/", exist_ok=True)
now = datetime.now()
log_location = f"logs/run_heuristic/{now.strftime('%d_%m_%Y_%H_%M_%S')}/"
os.makedirs("logs/run_heuristic/", exist_ok=True)
os.makedirs(log_location, exist_ok=True)
env_kwargs = {}
env_kwargs['spawn_mode'] = "corner"
env_kwargs['spawn_range'] = 350
env_kwargs['helicopter_battery_life'] = 200
env_kwargs['helicopter_recharge_time'] = 40
env_kwargs['num_search_parties'] = 5
# env_kwargs['reward_scheme'] = reward_scheme
env_kwargs['random_cameras'] = False
env_kwargs['observation_step_type'] = "Fugitive"
env_kwargs['debug'] = False
env_kwargs['observation_terrain_feature']=False
env_kwargs['stopping_condition'] = True

# Directory to randomly cycle between all the maps
# env_kwargs['terrain_map'] = 'simulator/forest_coverage/maps'

# Single map to always test on one map
env_kwargs['terrain_map'] = 'simulator/forest_coverage/maps_0.2/1.npy'
env_kwargs['camera_file_path'] = "simulator/camera_locations/original_and_more.txt"
env = SmugglerEnv(**env_kwargs)

num_iter = 1

show = False
mean_return = 0.0
for seed in range(0, 50):

    # seed = 21 # evasion a little weird on this seed
    np.random.seed(seed)
    random.seed(seed)
    imgs = []

    policy = HeuristicPolicy(env)
    observation = env.reset()
    done = False
    episode_return = 0.0
    i = 0
    imgs = []
    while not done:
        i += 1
        action = policy.predict(observation)
        observation, reward, done, _ = env.step(action[0])
        episode_return += reward

        if show:

            game_img = env.render('Policy', show=True, fast=True)

            blue_plan_img = cv2.imread("logs/temp/debug_plan.png")  # the blue heuristic plan debug figure
            # combine both images to one
            max_x = max(game_img.shape[0], blue_plan_img.shape[0])
            max_y = max(game_img.shape[1], blue_plan_img.shape[1])
            game_img_reshaped = np.pad(game_img, ((0, max_x - game_img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)
            blue_plan_reshaped = np.pad(blue_plan_img, ((0, max_x - blue_plan_img.shape[0]), (0, 0), (0, 0)), 'constant', constant_values=0)

            img = np.concatenate((game_img_reshaped, blue_plan_reshaped), axis=1)
            imgs.append(img)

        if done:
            print(action)
            # hideout = env.near_hideout()
            # print(env.timesteps, hideout.known_to_good_guys)
            break

    # save_video(imgs, log_location + "%d.mp4" % seed, fps=10)

    # mean_return += episode_return / num_iter
    # print(episode_return)
