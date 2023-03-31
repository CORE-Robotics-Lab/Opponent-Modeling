from simulator.prisoner_env import PrisonerEnv
import numpy as np
import random
""" We want to show that with two different initializations with the exact same setup except the unknown hideout location, the reults will show"""

def intialize_multimodal_env(observation_step_type="Fugitive"):
    """ Return a tuple of two environments """


    terrain_map = f'simulator/forest_coverage/map_set/0.npy'
    mountain_locations = [(400, 300), (1600, 1800)] # original mountain setup
    camera_configuration="simulator/camera_locations/original_and_more.txt"
    terrain=None
    step_reset = True
    
    # known_hideout_locations = [[323, 1623], [1804, 737], [317, 2028], [819, 1615], [1145, 182], [1304, 624], [234, 171], [2398, 434], [633, 2136], [1590, 2]],
    # unknown_hideout_locations = [[376, 1190], [909, 510], [397, 798], [2059, 541], [2011, 103], [901, 883], [1077, 1445], [602, 372], [80, 2274], [279, 477]],

    seed = 0
    known_hideout_locations = [[323, 1623]]
    unknown_hideout_locations = [[80, 2274], [279, 477]]

    env_one = PrisonerEnv(terrain=terrain,
                      spawn_mode='normal', #set position to be in the exact same place
                      observation_step_type=observation_step_type,
                      random_cameras=False,
                      camera_file_path=camera_configuration,
                      mountain_locations=mountain_locations,
                      camera_range_factor=1.0,
                      observation_terrain_feature=False,
                      random_hideout_locations=False,
                      spawn_range=350,
                      helicopter_battery_life=200,
                      helicopter_recharge_time=40,
                      num_search_parties=5,
                      terrain_map=terrain_map,
                      step_reset = step_reset,
                      known_hideout_locations=known_hideout_locations,
                      unknown_hideout_locations=unknown_hideout_locations
                      )
    env_one.seed(seed)

    known_hideout_locations = [[323, 1623]]
    unknown_hideout_locations = [[2011, 103], [279, 477]]

    env_two = PrisonerEnv(terrain=terrain,
                      spawn_mode='corner',
                      observation_step_type=observation_step_type,
                      random_cameras=False,
                      camera_file_path=camera_configuration,
                      mountain_locations=mountain_locations,
                      camera_range_factor=1.0,
                      observation_terrain_feature=False,
                      random_hideout_locations=False,
                      spawn_range=350,
                      helicopter_battery_life=200,
                      helicopter_recharge_time=40,
                      num_search_parties=5,
                      terrain_map=terrain_map,
                      step_reset = step_reset,
                      known_hideout_locations=known_hideout_locations,
                      unknown_hideout_locations=unknown_hideout_locations
                      )
    env_two.seed(seed)

    return env_one, env_two