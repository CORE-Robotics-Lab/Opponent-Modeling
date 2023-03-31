"""
Wrapper to support LSTM sequence.
"""

import gym
import numpy as np
import copy
import gc

from simulator import PrisonerEnv

class PrisonerSequenceEnv(gym.Wrapper):
    """ Batches the observations for an lstm """
    def __init__(self,
                 env: PrisonerEnv,
                 sequence_len=1):
        """
        :param env: the PrisonerEnv instance to wrap.
        :param sequence_len: number of stacked observations to feed into the LSTM
        :param deterministic: whether the worker(s) should be run deterministically
        """
        super().__init__(env)
        self.sequence_len = sequence_len

        # We store the most recent observation at the end of this array
        self.observation_array = np.zeros((sequence_len,) + self.observation_space.shape)
        
        self.env = env
        self.prediction_observation_array = np.zeros((sequence_len,) + self.env.prediction_observation_space.shape)
    def reset(self):
        obs = self.env.reset()
        self.observation_array = np.zeros((self.sequence_len,) + self.observation_space.shape)
        self.observation_array[-1] = obs
        return self.observation_array

    def step(self, action):
        obs, reward, done, i = self.env.step(action)
        self.observation_array[0:-1] = self.observation_array[1:]
        self.observation_array[-1] = obs

        self.prediction_observation_array[0:-1] = self.prediction_observation_array[1:]
        self.prediction_observation_array[-1] = self.env._prediction_observation
        
        return self.observation_array, reward, done, i

    def get_state(self):
        """
        Compile a dictionary to represent environment's current state (only including things that will change in .step())
        :return: a dictionary with prisoner_location, search_party_locations, helicopter_locations, timestep, done, prisoner_location_history, is_detected
        """
        prisoner_location = self.prisoner.location.copy()
        search_party_locations = []
        search_party_plans = []
        for search_party in self.env.search_parties_list:
            search_party_locations.append(search_party.location.copy())
            search_party_plans.append(search_party.planned_path.copy())
        helicopter_locations = []
        helicopter_plans = []
        for helicopter in self.env.helicopters_list:
            helicopter_locations.append(helicopter.location.copy())
            helicopter_plans.append(helicopter.planned_path.copy())
        timestep = self.env.timesteps
        done = self.env.done
        prisoner_location_history = self.env.prisoner_location_history.copy()
        is_detected = self.env.is_detected

        prediction_observation =  self.env._prediction_observation.copy()
        fugitive_observation = self.env._fugitive_observation.copy()
        ground_truth_observation =  self.env._ground_truth_observation.copy()
        blue_observation = self.env._blue_observation.copy()
        observation_array = self.observation_array.copy()

        # print(self.search_parties_list[0].location)
        return {
            "prisoner_location": prisoner_location,
            "search_party_locations": search_party_locations,
            "search_party_plans": search_party_plans, 
            "helicopter_locations": helicopter_locations,
            "helicopter_plans" : helicopter_plans, 
            "timestep": timestep,
            "done": done,
            "prisoner_location_history": prisoner_location_history,
            "is_detected": is_detected,
            # "blue_heuristic": copy.deepcopy(self.blue_heuristic),
            "prediction_observation": prediction_observation,
            "fugitive_observation": fugitive_observation,
            "ground_truth_observation": ground_truth_observation,
            "blue_observation": blue_observation,
            "observation_array": observation_array,
            "done": self.done,
        }

    def set_state(self, state_dict):
        """
        Set the state of the env by state_dict. Paired with `get_state`
        :param state_dict: a state dict returned by `get_state`
        """
        self.env.prisoner.location = state_dict["prisoner_location"].copy()
        for i, search_party in enumerate(self.env.search_parties_list):
            search_party.location = state_dict["search_party_locations"][i].copy()
            search_party.planned_path = state_dict["search_party_plans"][i].copy()
        for i, helicopter in enumerate(self.env.helicopters_list):
            helicopter.location = state_dict["helicopter_locations"][i].copy()
            helicopter.planned_path = state_dict["helicopter_plans"][i].copy()
        self.env.timesteps = state_dict["timestep"]
        self.env.done = state_dict["done"]
        self.env.prisoner_location_history = state_dict["prisoner_location_history"].copy()
        self.env.is_detected = state_dict["is_detected"]
        # self.blue_heuristic = state_dict["blue_heuristic"]

        self.env.search_parties_list = self.blue_heuristic.search_parties
        self.env.helicopters_list = self.blue_heuristic.helicopters

        # set previous observations
        self.env._prediction_observation = state_dict["prediction_observation"].copy()
        self.env._fugitive_observation = state_dict["fugitive_observation"].copy()
        self.env._ground_truth_observation = state_dict["ground_truth_observation"].copy()
        self.env._blue_observation = state_dict["blue_observation"].copy()
        self.done = state_dict["done"]
        self.observation_array = state_dict["observation_array"].copy()
        gc.collect()

    def set_state_list(self, state_list):
        """ Same function as set state but the state_list is [state_dict], used as a workaround for vectorizing the environment """
        self.set_state(state_list[0])

    def generate_policy_heatmap(self, current_state, policy, num_timesteps=2500, num_rollouts=20, end=False):
        self.env.generate_policy_heatmap(current_state, policy, num_timesteps=2500, num_rollouts=20, end=False)