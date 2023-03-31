import sys, os
sys.path.append(os.getcwd())
from Smuggler.simulator import SmugglerBothEnv

# Load environment from file
def load_environment(config_path):
    import yaml
    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    mountain_locs = []
    mountain_locs = [list(map(int, (x.split(',')))) for x in data['mountain_locations']]

    known_hideout_locations = [list(map(int, (x.split(',')))) for x in data['known_hideout_locations']]
    unknown_hideout_locations = [list(map(int, (x.split(',')))) for x in data['unknown_hideout_locations']]

    env = SmugglerBothEnv(**data, 
                          mountain_locations=mountain_locs, 
                          unknown_hideout_locations=unknown_hideout_locations, 
                          known_hideout_locations=known_hideout_locations)
    return env

if __name__ == "__main__":
    env = load_environment("simulator/configs/fixed_cams_random_uniform_start_camera_net.yaml")
