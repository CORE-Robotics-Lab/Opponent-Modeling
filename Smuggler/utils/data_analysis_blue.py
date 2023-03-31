import pickle
import numpy as np
import matplotlib.pyplot as plt

import os, sys
sys.path.append(os.getcwd())

from simulator.load_environment import load_environment

path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations/train_vector.npz"

env = load_environment("simulator/configs/fixed_cams_random_uniform_start_camera_net.yaml")

buffer = np.load(path, allow_pickle=True)

eps_length = 0
episode_lengths = []
found_num = 0

num_per_run = 0
helicopter_find = 0
run_finds = []

num_wins = 0
num_losses = 0

num_search_party_finds = []

print("Total number of timesteps: ", len(buffer["dones"]))

for done, obs in zip(buffer["dones"], buffer["blue_observations"]):
    if done == 1:
        # if rewards[0] > 0:
        #     num_wins += 1
        # elif rewards[0] < 0:
        #     num_losses += 1
        # else:
        #     print("Undefined final condition")

        episode_lengths.append(eps_length)
        run_finds.append(num_per_run)
        num_per_run = 0
        eps_length = 0
    else:
        eps_length += 1

    blue_obs = env.blue_obs_names.wrap(obs)
    found = False
    for i in range(env.num_known_cameras):
        found = found or (blue_obs['known_camera_%d' % i][0] == 1.) 

    for i in range(env.num_unknown_cameras):
        found = found or (blue_obs['unknown_camera_%d' % i][0] == 1.) 

    for i in range(env.num_helicopters):
        helicopter_find += (blue_obs['helicopter_%d' % i][0] == 1.)
        found = found or (blue_obs['helicopter_%d' % i][0] == 1.) 

    num_sp_find = 0
    for i in range(env.num_search_parties):
        # print(blue_obs['search_party_%d' % i])
        num_sp_find += (blue_obs['search_party_%d' % i][0] == 1.)
        found = found or (blue_obs['search_party_%d' % i][0] == 1.) 

    num_search_party_finds.append(num_sp_find)

    found_num += found
    num_per_run += found

print(f"Number of wins: {num_wins}, Number of losses: {num_losses}")

# num search party finds
count_arr = np.bincount(np.array(num_search_party_finds))
print(count_arr)
for i in range(0, 6):
    print(f"Number of timesteps {i} search parties detect the fugitive: {count_arr[i]}, {count_arr[i] / len(buffer['dones']) * 100}%")

print(f"Number of timesteps the helicopter found the fugitive: {helicopter_find}, {helicopter_find / len(buffer['dones']) * 100}%")

print(f"Total number of timesteps the fugitive was found: {found_num}, {found_num / len(buffer['dones']) * 100}%")
# print(run_finds)

# save list to csv
np.savetxt("run_finds.csv", run_finds, delimiter=",")
np.savetxt("episode_lengths.csv", episode_lengths, delimiter=",")

# n, bins, patches = plt.hist(episode_lengths, facecolor='blue', alpha=0.5)
# # save plot to file
# plt.savefig('episode_lengths.png')

n, bins, patches = plt.hist(run_finds, facecolor='blue', alpha=0.5)
plt.xlabel('Number of any Blue Detections of Red per run')
plt.ylabel('Number of runs')

# print(np.sum(run_finds))

# save plot to file
plt.savefig('temp/run_finds_filtering.png')
