import matplotlib.pyplot as plt
import time
import numpy as np


class BlueHeuristic:
    def __init__(self, env, debug=False):
        self.env = env
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []
        self.debug = debug
        if debug:
            self.fig, self.ax = plt.subplots()
            plt.close(self.fig)
            plt.clf()

    def reset(self):
        # self.search_parties = search_parties
        # self.helicopters = helicopters
        self.detection_history = []

    def predict(self, blue_observation: np.ndarray):
        """ Inform the heuristic about the observation of the blue agents. """
        blue_obs_names = self.env.blue_obs_names
        wrapped_blue_observation = blue_obs_names(blue_observation)
        new_detection = wrapped_blue_observation["prisoner_detected"]
        # check if new_detection equals [-1, -1]
        # print("new_detection: ", new_detection)
        if np.array_equiv(new_detection, np.array([-1, -1])):
            new_detection = None
        else:
            new_detection = (new_detection*2428).tolist()
        return self.step(new_detection)

    def step(self, new_detection):
        if new_detection is not None:
            self.detection_history.append((new_detection, self.timesteps))
            if len(self.detection_history) == 1:
                self.command_each_party("plan_path_to_loc", new_detection)
            else:
                vector = np.array(new_detection) - np.array(self.detection_history[-2][0])
                speed = np.sqrt(np.sum(np.square(vector))) / (self.timesteps - self.detection_history[-2][1])
                direction = np.arctan2(vector[1], vector[0])
                self.command_each_party("plan_path_to_intercept", speed, direction, new_detection)
        if self.debug:
            self.debug_plot_plans()
        # self.command_each_party("move_according_to_plan")
        # instead of commanding each party to move, grab the action that we pass into the environment

        # self.command_each_party("get_action_according_to_plan
        return self.get_each_action()

    def get_each_action(self):
        # get the action for each party
        actions = []
        for search_party in self.env.search_parties_list:
            action = np.array(search_party.get_action_according_to_plan())
            actions.append(action)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                action = np.array(helicopter.get_action_according_to_plan())
                actions.append(action)
        else:
            for helicopter in self.env.helicopters_list:
                action = np.array([0, 0, 0])
                actions.append(action)
        return actions

    @property
    def timesteps(self):
        return self.env.timesteps

    def init_pos(self):
        # a heuristic strategy to initialize position of each blue?
        pass

    def init_behavior(self):
        # initialize the behavior at the beginning before any detection is made?
        self.command_each_party("plan_path_to_random")

    def command_each_party(self, command, *args, **kwargs):
        for search_party in self.env.search_parties_list:
            getattr(search_party, command)(*args, **kwargs)
        if self.env.is_helicopter_operating():
            for helicopter in self.env.helicopters_list:
                getattr(helicopter, command)(*args, **kwargs)

    def debug_plot_arrow(self, from_x, from_y, to_x, to_y, **kwargs):
        self.ax.arrow(from_x, from_y, to_x - from_x, to_y - from_y, **kwargs)

    def debug_plot_plans(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([0, 2428])
        self.ax.set_ylim([0, 2428])
        self.ax.set_aspect('equal')

        all_blue = self.search_parties + self.helicopters if self.env.is_helicopter_operating() else self.search_parties

        for i in range(len(self.detection_history) - 1):
            self.debug_plot_arrow(self.detection_history[i][0][0], self.detection_history[i][0][1],
                                  self.detection_history[i+1][0][0], self.detection_history[i+1][0][1],
                                  color='red', head_width=10)

        if len(self.detection_history) > 1:
            print("second_last_detection:", self.detection_history[-2])
        if len(self.detection_history) > 0:
            print("last_detection:", self.detection_history[-1])

        for blue_agent in all_blue:
            planned_path = blue_agent.planned_path
            current_loc = blue_agent.location
            for plan in planned_path:
                if self.debug:
                    print(blue_agent, "loc:", blue_agent.location, "plan:", plan)
                if plan[0] == 'l':
                    self.debug_plot_arrow(current_loc[0], current_loc[1],
                                          plan[1], plan[2], color='black', head_width=20)
                    current_loc = (plan[1], plan[2])
                elif plan[0] == 'd':
                    length_of_direction = 50
                    self.ax.arrow(current_loc[0], current_loc[1],
                                  plan[1] * length_of_direction, plan[2] * length_of_direction,
                                  color='pink', head_width=20)
                else:
                    self.debug_plot_arrow(current_loc[0], current_loc[1],
                                          plan[1], plan[2], color='orange', head_width=20)
                    current_loc = (plan[1], plan[2])
        plt.savefig("logs/temp/debug_plan_%d.png" % self.timesteps)
        plt.savefig("logs/temp/debug_plan.png")
        plt.close(self.fig)
        plt.clf()
        # input("Enter to continue")
