""" Script to generate images for 2/15 presentation"""

import numpy as np
import torch

from utils import evaluate_mean_reward
from simulator import SmugglerEnv
from fugitive_policies.heuristic import HeuristicPolicy

import cv2

#set seed
np.random.seed(0)
torch.manual_seed(0)

env_kwargs = {}
env_kwargs['spawn_mode'] = "uniform"
# env_kwargs['reward_scheme'] = reward_scheme
env_kwargs['random_cameras'] = False
env_kwargs['observation_step_type'] = "Fugitive"
env_kwargs['terrain_map_file'] = None
env_kwargs['camera_file_path'] = "simulator/camera_locations/fill_camera_locations.txt"
env = SmugglerEnv(**env_kwargs)

policy = HeuristicPolicy(env)

observation = env.reset()

env.prisoner.location = [1400, 2000]

img = env.render('Policy', show=False, fast=False)

#save image
cv2.imwrite('test.png', img)

# done = False
# episode_return = 0.0
# # while not done:
# action = policy(observation)
# observation, reward, done, _ = env.step(action[0])
# episode_return += reward

    # if done:
    #     break
