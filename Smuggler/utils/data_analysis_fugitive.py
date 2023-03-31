import numpy as np
import math

import os, sys
from simulator.observation_spaces import ObservationNames
sys.path.append(os.getcwd())

# path = "buffers/msp/train_300.npz"
# path = "buffers/msp_prediction_epsilon/train_300_maps.npz"

map_num = 0
# path = f"/nethome/sye40/PrisonerEscape/buffers_msp/map_{map_num}_run_300_eps_0.npz"

# path = f"buffers_msp/map_0_run_300_eps_0.npz"
# path = "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_run_300_rrt.npz"

# path =  "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_400_RRT_gnn_save.npz"
# path = "/nethome/sye40/PrisonerEscape/datasets/post_red_obs_fix/map_0_run_300_heuristic_eps_0.1.npz"

path = "/nethome/sye40/PrisonerEscape/datasets/random_start_locations/train_vector.npz"

num_search_parties = 5
num_helicopters = 1
max_timesteps = 4320

####### Original Hideout Locations ########
# known_hideout_locations = [[323, 1623], [1804, 737], [317, 2028], [819, 1615], [1145, 182], [1304, 624], [234, 171], [2398, 434], [633, 2136], [1590, 2]]
# unknown_hideout_locations = [[376, 1190], [909, 510], [397, 798], [2059, 541], [2011, 103], [901, 883], [1077, 1445], [602, 372], [80, 2274], [279, 477]]

###### New Hideout Locations #########
known_hideout_locations = [(2077, 2151), (2170, 603), (37, 1293), (1890, 30), (1151, 2369), (356, 78), (1751, 1433), (1638, 1028), (1482, 387), (457, 1221)]
unknown_hideout_locations = [(234, 2082), (1191, 950), (563, 750), (2314, 86), (1119, 1623), (1636, 2136), (602, 1781), (2276, 1007), (980, 118), (2258, 1598)]

data = np.load(path, allow_pickle=True)
buffer = data['red_observations']
locations = data['red_locations']
obs_names = data['prediction_dict'].item()
total_length = buffer.shape[0]

print("Total number of sample: ", buffer.shape[0])
#  = env.fugitive_obs_names

# Helicopter stats
idx_helicopter = obs_names["helicopter_detect_0"]
helo_0 = buffer[:,idx_helicopter[0]:idx_helicopter[1]]
print(f"Number of timesteps fugitive detects helicopter: {np.sum(helo_0[:, 0])}")

# Search Parties
search_party_list = []
for i in range(num_search_parties):
    string_identifier = f"search_party_detect_{i}"
    idx_search_party = obs_names[string_identifier]
    search_party_list.append(buffer[:,idx_search_party[0]])

summed_search_party = np.stack(search_party_list, axis=1).sum(axis=1)
count_arr = np.bincount(summed_search_party.astype(int))
percentage_detect_save = []
for i in range(0, 6):
    print(f"Number of timesteps fugitive detects {i} search party: {count_arr[i]}")
    percentage_detect_save.append(str(count_arr[i]/total_length))
print(",".join(percentage_detect_save))

# Identify winrate

def near_hideout(prisoner_location, hideout_list, hideout_radius):
    for hideout in hideout_list:
        if ((np.asarray(hideout) - np.asarray(prisoner_location))**2).sum()**.5 <= hideout_radius+1e-6:
            # print(f"Reached a hideout that is {hideout.known_to_good_guys} known to good guys")
            return 1
    return 0

num_episodes = 1
last_timestep = buffer[0][0]
last_location = locations[0]

num_wins = 0
num_losses = 0

# print(env.hideout_radius)

# known_hideout_locations = [[323, 1623], [1804, 737], [317, 2028], [819, 1615], [1145, 182], [1304, 624], [234, 171], [2398, 434], [633, 2136], [1590, 2]],
# unknown_hideout_locations = [[376, 1190], [909, 510], [397, 798], [2059, 541], [2011, 103], [901, 883], [1077, 1445], [602, 372], [80, 2274], [279, 477]],

run_length = 0
num_per_run = 0
helicopter_find = 0
found_num = 0
run_finds = []
percentage_finds = []
run_lengths = []


num_search_party_finds = []
for obs, location in zip(buffer[1:], locations[1:]):
    # fugitive_obs = obs_names.wrap(obs)
    fugitive_obs = ObservationNames.NamedObservation(obs, obs_names)
    timestep = obs[0]
    if not math.isclose(timestep, last_timestep + 1/max_timesteps):
        # End of episode
        prisoner_location = tuple((last_location * 2428).astype(int))

        num_wins += near_hideout(prisoner_location, unknown_hideout_locations, 60)
        num_losses += near_hideout(prisoner_location, known_hideout_locations, 70)
        num_episodes += 1
        # print(prisoner_location, num_wins)
        run_finds.append(num_per_run)
        percentage_finds.append(100*num_per_run/run_length)
        run_lengths.append(run_length)
        num_per_run = 0
        run_length = 0


    found = False
    for i in range(num_helicopters):
        found = found or (fugitive_obs['helicopter_detect_%d' % i][0] == 1.) 
        helicopter_find += (fugitive_obs['helicopter_detect_%d' % i][0] == 1.)

    num_sp_find = 0
    for i in range(num_search_parties):
        num_sp_find += (fugitive_obs['search_party_detect_%d' % i][0] == 1.)
        found = found or (fugitive_obs['search_party_detect_%d' % i][0] == 1.) 
    num_search_party_finds.append(num_sp_find)
    found_num += found
    num_per_run += found
    run_length += 1

    last_location = location
    last_timestep = timestep

# num search party finds
count_arr = np.bincount(np.array(num_search_party_finds))
percentage_detect_save = []
for i in range(0, 6):
    print(f"Number of timesteps {i} search parties detect the fugitive: {count_arr[i]/total_length:4.5f}, {count_arr[i]}")
    percentage_detect_save.append(str(count_arr[i]/total_length))
print(",".join(percentage_detect_save))
print("Timesteps observed by helicopter: ", helicopter_find/total_length)
print("Total timesteps found fugitive: ", found_num/total_length)
# print(f"Number of detections in a run"  % {run_finds})
# print(run_finds)
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(run_finds, facecolor='blue', alpha=0.5)
plt.xlabel('Number of Red Detections of any Blue agent per run')
plt.ylabel('Number of runs')
plt.savefig(f'temp/stats_prediction/map_{map_num}_run_finds.png')

# print(percentage_finds)
plt.figure()
n, bins, patches = plt.hist(percentage_finds, facecolor='blue', alpha=0.5)
plt.xlabel('Percentage Red Detections of any Blue agent per run')
plt.ylabel('Number of runs')
plt.savefig(f'temp/stats_prediction/map_{map_num}_percentage_run_prediction.png')
# plt.close()

plt.figure()
n, bins, patches = plt.hist(run_lengths, facecolor='blue', alpha=0.5)
plt.xlabel('Run Length')
plt.ylabel('Number of runs')
plt.savefig(f'temp/stats_prediction/map_{map_num}_run_length.png')


print(num_wins, num_losses)