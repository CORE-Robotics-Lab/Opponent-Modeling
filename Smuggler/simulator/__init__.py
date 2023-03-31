# This file exports PrisonerEscape environment by registering in OpenAI Gym
# so that we could do `gym.make('PrisonerEscape-v0')`


from gym.envs import register
from simulator.smuggler_env import SmugglerBothEnv, RewardScheme
from simulator.smuggler_perspective_envs import SmugglerBlueEnv
from simulator.observation_spaces import ObservationNames
from simulator.terrain import Terrain
# from simulator.prisoner_env_variations import initialize_prisoner_environment

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return

    print("Registering custom gym environments")
    register(id='PrisonerEscape-v0', entry_point='simulator.prisoner_env:PrisonerEnv', kwargs={})

    _REGISTERED = True


register_custom_envs()