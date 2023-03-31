import sys, os
sys.path.append(os.getcwd())
from Prison_Escape.environment.prisoner_perspective_envs import PrisonerEnv
from Prison_Escape.environment import PrisonerBothEnv, PrisonerBlueEnv, PrisonerEnv
from Prison_Escape.blue_policies.heuristic import BlueHeuristic
from Prison_Escape.fugitive_policies.heuristic import HeuristicPolicy
from Prison_Escape.fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
import numpy as np
import random

# Load environment from file
def load_environment(config_path):
    import yaml
    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    mountain_locs = []
    mountain_locs = [list(map(int, (x.split(',')))) for x in data['mountain_locations']]

    known_hideout_locations = [list(map(int, (x.split(',')))) for x in data['known_hideout_locations']]
    unknown_hideout_locations = [list(map(int, (x.split(',')))) for x in data['unknown_hideout_locations']]

    env = PrisonerBothEnv(
                      terrain=data['terrain'],
                      spawn_mode=data['spawn_mode'],
                      min_distance_from_hideout_to_start=data['min_distance_from_hideout_to_start'],
                    #   observation_step_type=data['observation_step_type'],
                      random_cameras=data['random_cameras'],
                      camera_file_path=data['camera_file_path'],
                      mountain_locations=mountain_locs,
                      camera_range_factor=data['camera_range_factor'],
                      observation_terrain_feature=data['observation_terrain_feature'],
                      random_hideout_locations=data['random_hideout_locations'],
                      spawn_range=data['spawn_range'],
                      helicopter_battery_life=data['helicopter_battery_life'],
                      helicopter_recharge_time=data['helicopter_recharge_time'],
                      num_search_parties=data['num_search_parties'],
                      num_helicopters = data['num_helicopters'],
                      terrain_map=data['terrain_map'],
                      step_reset = data['step_reset'],
                      camera_net_bool=data['camera_net_bool'],
                      include_camera_at_start=data['include_camera_at_start'],
                      include_start_location_blue_obs=data['include_start_location_blue_obs'],
                      num_random_known_cameras = data['num_random_known_cameras'],
                      num_random_unknown_cameras = data['num_random_unknown_cameras'],
                      unknown_hideout_locations=unknown_hideout_locations,
                      known_hideout_locations=known_hideout_locations,
                      store_last_k_fugitive_detections = data['store_last_k_fugitive_detections'],
                      search_party_speed = data['search_party_speed'],
                      helicopter_speed = data['helicopter_speed'],
                      detection_factor=data['detection_factor'],
                      )
    return env

if __name__ == "__main__":
    env = load_environment("configs/fixed_cams_random_uniform_start_camera_net.yaml")
