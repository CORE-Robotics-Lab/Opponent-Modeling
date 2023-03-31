from simulator.smuggler_env import SmugglerBothEnv, ObservationType
from blue_policies.heuristic import BlueHeuristic
from fugitive_policies.rrt_star_adversarial_avoid import RRTStarAdversarialAvoid
from fugitive_policies.a_star_policy import AStarPolicy
import numpy as np
import gym

class SmugglerEnv(gym.Wrapper):
    """ Produce environment to match our previous implementation to hot swap in
    
    This environment returns red observations and takes in red actions
    """
    def __init__(self,
                 env: SmugglerBothEnv,
                 blue_policy: None):
        super().__init__(env)
        self.env = env
        self.blue_policy = blue_policy
        # # ensure the environment was initialized with blue observation type
        # assert self.env.observation_type == ObservationType.Blue
        self.observation_space = self.env.fugitive_observation_space
        self.obs_names = self.env.fugitive_obs_names

    def reset(self, seed=None):
        self.env.reset(seed)
        if type(self.blue_policy) == BlueHeuristic:
            self.blue_policy.reset()
            self.blue_policy.init_behavior()
        return self.env._fugitive_observation
        
    def step(self, red_action):
        # get red observation for policy
        blue_obs_in = self.env._blue_observation
        blue_action = self.blue_policy.predict(blue_obs_in)
        red_obs, _, reward, done, i = self.env.step_both(red_action, blue_action)
        return red_obs, reward, done, i 

class SmugglerBlueEnv(gym.Wrapper):
    """ This environment return blue observations and takes in blue actions """
    def __init__(self,
                 env: SmugglerBothEnv,
                 fugitive_policy):
        super().__init__(env)
        self.env = env
        self.fugitive_policy = fugitive_policy

        # # ensure the environment was initialized with blue observation type
        # assert self.env.observation_type == ObservationType.Blue
        self.observation_space = self.env.blue_observation_space
        self.obs_names = self.env.blue_obs_names

    def reset(self, seed=None):
        self.env.reset(seed)
        if type(self.fugitive_policy) == RRTStarAdversarialAvoid:
            self.fugitive_policy.reset()

        if type(self.fugitive_policy) == AStarPolicy:
            self.fugitive_policy.reset()
        return self.env._blue_observation
        
    def step(self, blue_action):

        # get red observation for policy
        red_obs_in = self.env._fugitive_observation
        red_action = self.fugitive_policy.predict(red_obs_in)
        _, blue_obs, reward, done, i = self.env.step_both(red_action[0], blue_action)
        return blue_obs, reward, done, i 