# This file defines several abstract classes to define common behaviors of objects in the environment
import numpy as np
import matplotlib.pyplot as plt
from Prison_Escape.environment.utils import distance

class AbstractObject:
    def __init__(self, terrain, location):
        """
        Any abstract object with a location on the terrain.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        """
        self.terrain = terrain
        self.location = location

    @staticmethod
    def generate_random_locations(dim_x, dim_y):
        """
        Helper function to generate a random location from [0, dim_x], [0, dim_y]
        :param dim_x: the maximum coordinate on x
        :param dim_y: the maximum coordinate on y
        :return: a list of length 2 representing the location
        """
        return AbstractObject.generate_random_locations_with_range((0, dim_x), (0, dim_y))

    @staticmethod
    def generate_random_locations_with_range(range_x, range_y):
        """
        Helper funciton to generate a random location from range_x and range_y
        :param range_x: the range of coordinates on x-axis. Could be a list or tuple of length 2 representing the range.
        :param range_y: the range of coordinates on y-axis. Could be a list or tuple of length 2 representing the range.
        :return: a list of length 2 representing the location
        """
        x_coord = np.random.randint(range_x[0], range_x[1])
        y_coord = np.random.randint(range_y[0], range_y[1])
        return [x_coord, y_coord]


class MovingObject(AbstractObject):
    def __init__(self, terrain, location, speed):
        """
        MovingObject defines object that could move.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        :param speed: a number representing speed with unit in grids
        """
        AbstractObject.__init__(self, terrain, location)
        self.speed = speed
        self.direction_map = {
            0: np.array([0, 1]),  # up
            1: np.array([1, 0]),  # right
            2: np.array([0, -1]),  # down
            3: np.array([-1, 0])  # left
        }
        self.destination = self.location
        self.planned_path = []
        self.debug = False

    def move_random(self):
        """
        Defines the movement of the object
        """
        direction = self.direction_map[np.random.randint(0, 4)]
        old_location = self.location.copy()
        self.location += direction * self.speed
        if self.terrain.violate_edge_constraints(self.location[0], self.location[1], 1, 1):
            self.location = old_location

    def move(self, camera_list, time_step_delta, last_known_fugitive_location):
        """
            Movement heuristic when the helicopter or search party has not been detected (logic exists in prisoner_env.env.step)
            :param camera_list: a list of camera objects
            :param time_step_delta: time between current timestep and last detected timestep
            :param last_known_fugitive_location: the last known fugitive location (np.array[x, y])
        """ 
        if np.array_equal(self.location,self.destination):
            # If we arrive at the destination, choose a new destination 
            self.destination = self.sample_destinations(time_step_delta,last_known_fugitive_location)
        self.path(camera_list)

    def transform_destination_to_direction(self, destination):
        path_vector = np.array([destination[0] - self.location[0], destination[1] - self.location[1]])
        distance_movement = np.sqrt(path_vector[0] ** 2 + path_vector[1] ** 2)
        if distance_movement == 0:
            direction = np.zeros_like(path_vector)
        else:
            direction = path_vector / distance_movement
        # print(direction[0]**2 + direction[1]**2)
        # speed = min(distance_movement, speed)
        return direction

    def get_action_according_to_plan(self):
        # action is (x, y, velocity)
        current_movement = self.planned_path[0]
        action = None
        if current_movement[0] == 'l':
            # self.path_v2(destination=(current_movement[1], current_movement[2]))

            destination = np.array([current_movement[1], current_movement[2]], dtype=np.int)
            sample_radius = 200
            while self.in_range_of_mountain(destination, 161)[1]:
                destination = np.array([np.random.randint(destination[0] - sample_radius, destination[0] + sample_radius),
                                            np.random.randint(destination[1] - sample_radius, destination[1] + sample_radius)],
                                            dtype=np.int)
                self.planned_path.pop(0)
                self.planned_path = [('l', destination[0], destination[1])] + self.planned_path

            direction = self.transform_destination_to_direction(destination)
            speed = min(distance(self.location, [current_movement[1], current_movement[2]]), self.speed)
            action = (direction[0], direction[1], speed)
            if distance(self.location, (current_movement[1], current_movement[2])) <= 10:
                self.planned_path.pop(0)
                if len(self.planned_path) == 0:
                    # move to loc just finished, start a spiral
                    self.plan_spiral()
        elif current_movement[0] == 'ls':
            # self.path_v2(destination=(current_movement[1], current_movement[2]),
            #              constraint_speed=30, mountain_outer_range=150)  # spiral detection range=22, and make mountain behavior more conservative
            destination = np.array([current_movement[1], current_movement[2]], dtype=np.int)
            sample_radius = 200
            while self.in_range_of_mountain(destination, 161)[1]:
                destination = np.array([np.random.randint(destination[0] - sample_radius, destination[0] + sample_radius),
                                            np.random.randint(destination[1] - sample_radius, destination[1] + sample_radius)],
                                            dtype=np.int)
                self.planned_path.pop(0)
                self.planned_path = [('l', destination[0], destination[1])] + self.planned_path
            # direction = self.transform_destination_to_direction(np.array([current_movement[1], current_movement[2]]))
            direction = self.transform_destination_to_direction(destination)
            speed = min(distance(self.location, [current_movement[1], current_movement[2]]), 30)
            action = (direction[0], direction[1], speed)
            if distance(self.location, (current_movement[1], current_movement[2])) <= 10:
                self.planned_path.pop(0)
                if len(self.planned_path) == 0:
                    # spiral just finished
                    self.plan_path_to_random()
        elif current_movement[0] == 'd':
            if current_movement[3] > 0:
                # fast moving mode
                # success = self.path_v2(direction=(current_movement[1], current_movement[2])) 
                # TODO: Sean - Is success always true?
                action = (current_movement[1], current_movement[2], self.speed)
                # if success: 
                self.planned_path[0] = (current_movement[0], current_movement[1], current_movement[2], current_movement[3] - 1, current_movement[4])
            else:
                # slow moving mode to detect carefully
                if current_movement[4] == 'c':
                    # chasing
                    # self.path_v2(direction=(current_movement[1], current_movement[2]), constraint_speed=22)
                    action = (current_movement[1], current_movement[2], 22)
                else:
                    # meeting
                    # self.path_v2(direction=(current_movement[1], current_movement[2]), constraint_speed=8)
                    action = (current_movement[1], current_movement[2], 8)
            if not ((self.speed < self.location[0] < self.terrain.dim_x - self.speed)
                    and (self.speed < self.location[1] < self.terrain.dim_y - self.speed)):
                # going along direction till the edge
                self.planned_path.pop(0)
                if len(self.planned_path) == 0:
                    self.plan_path_to_random()
        else:
            raise NotImplementedError
        return action

    def move_according_to_plan(self):
        current_movement = self.planned_path[0]
        if current_movement[0] == 'l':
            self.path_v2(destination=(current_movement[1], current_movement[2]))
            if distance(self.location, (current_movement[1], current_movement[2])) == 0:
                self.planned_path.pop(0)
                if len(self.planned_path) == 0:
                    # move to loc just finished, start a spiral
                    self.plan_spiral()
        elif current_movement[0] == 'ls':
            self.path_v2(destination=(current_movement[1], current_movement[2]),
                         constraint_speed=30, mountain_outer_range=150)  # spiral detection range=22, and make mountain behavior more conservative
            if distance(self.location, (current_movement[1], current_movement[2])) == 0:
                self.planned_path.pop(0)
                if len(self.planned_path) == 0:
                    # spiral just finished
                    self.plan_path_to_random()
        elif current_movement[0] == 'd':
            if current_movement[3] > 0:
                # fast moving mode
                success = self.path_v2(direction=(current_movement[1], current_movement[2]))
                if success:
                    self.planned_path[0] = (current_movement[0], current_movement[1], current_movement[2], current_movement[3] - 1, current_movement[4])
            else:
                # slow moving mode to detect carefully
                if current_movement[4] == 'c':
                    # chasing
                    self.path_v2(direction=(current_movement[1], current_movement[2]), constraint_speed=22)
                else:
                    # meeting
                    self.path_v2(direction=(current_movement[1], current_movement[2]), constraint_speed=8)
            if not ((self.speed < self.location[0] < self.terrain.dim_x - self.speed)
                    and (self.speed < self.location[1] < self.terrain.dim_y - self.speed)):
                # going along direction till the edge
                self.planned_path.pop(0)
                if len(self.planned_path) == 0:
                    self.plan_path_to_random()
        else:
            raise NotImplementedError

    def follow_fugitive(self, prisoner_location, camera_list):
        """ 
            movement heuristic when helicopter has been detected (logic exists in prisoner_env.env.step)
            currently cheats by giving helicopters omniscience once fugitive is detected once
        """
        self.destination = prisoner_location
        self.path(camera_list)

    def sample_destinations(self, time_step_delta=None, last_known_fugitive_location=None, sample_method = 'random'):
        """ Sample destination to go to """

        x = np.arange(0,2428)
        y = np.arange(0,2428)
        xx,yy = np.meshgrid(x,y,sparse=True)

        # V0 - RANDOM HEURISTIC
        if sample_method == 'random':
            sample_index = [np.random.randint(0, 2428),np.random.randint(0, 2428)]
            while self.terrain.world_representation[0, sample_index[0], sample_index[1]] == 1 :
                sample_index = [np.random.randint(0, 2428),np.random.randint(0, 2428)]
        # V1 - RBF HEURISTIC
        elif sample_method == 'rbf':
            # assumes uniform distribution of fugitive speeds
            #last_known_fugitive_location = np.array((0,0))
            #time_step_delta = 1000
            b = 14
            a = 1
            mu_s = (b+a)/2
            var_s = (b-a)**2 / 12 * np.sqrt(time_step_delta/4)
            r = mu_s * time_step_delta # very rough approximation (could be smarter about radius prediction)
            # create mesh grid and form rbf
            rbf = np.exp(-1/(2*var_s**2) * abs( r-np.sqrt( (xx-last_known_fugitive_location[0])**2 + (yy-last_known_fugitive_location[1])**2 ) )**2 ) # * 1/(var_s*np.sqrt(2*np.pi)) * 
            # bias towards where we think fugitive is more likely to head (hard coded towards (0,0) now, but this can be towards known hideouts or direction we previously observed fugitive moving in)
            gradient = np.outer(np.linspace(1,0,2428),np.linspace(1,0,2428))
            rbf_with_gradient = rbf * gradient + (0.0001*np.ones_like(rbf))
            rbf_with_gradient = rbf_with_gradient / sum(sum(rbf_with_gradient))

            # sample from distribution
            flat = rbf_with_gradient.flatten()
            sample_index = np.unravel_index(np.random.choice(a=flat.size, p=flat),rbf_with_gradient.shape)
            # if we sample a mountain, re-sample
            while self.terrain.world_representation[0, sample_index[0], sample_index[1]] == 1 :
                flat = rbf_with_gradient.flatten()
                sample_index = np.unravel_index(np.random.choice(a=flat.size, p=flat),rbf_with_gradient.shape)

        plot_flag = False
        if plot_flag:
            if sample_method == 'random':
                plt.contourf(x,y,np.zeros((2428,2428)))
            elif sample_method == 'rbf':
                plt.contourf(x,y,rbf_with_gradient)
            plt.plot(sample_index[0], sample_index[1], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
            plt.axis('scaled')
            plt.show()

        return sample_index

    def path_v2(self, destination=None, direction=None, mountain_inner_range=140, mountain_outer_range=160,
                constraint_speed=None):

        """
        Determine one step that the helicopter or search party will take to reach its destination
            :param destination: the desired location
            :param mountain_inner_range: the range at which the agent will move directly away from the mountain center
            :param mountain_outer_range: the range at which the agent will move perpendicular to the mountain
            :param constraint_speed: speed upper limit (used in spiral coverage)
        """
        direction_mode = False
        if direction:
            direction_mode = True
        speed = self.speed
        if constraint_speed:
            speed = min(constraint_speed, speed)

        # if destination command
        if destination is not None:
            path_vector = np.array([destination[0] - self.location[0], destination[1] - self.location[1]])
            distance_movement = np.sqrt(path_vector[0] ** 2 + path_vector[1] ** 2)
            if distance_movement == 0:
                direction = np.zeros_like(path_vector)
            else:
                direction = path_vector / distance_movement
            speed = min(distance_movement, speed)
        else:
            # destination = (np.array([direction[0], direction[1]]) * speed).astype(np.int) + self.location
            distance_movement = speed
        # destination = np.array(destination, dtype=np.int)
        direction = np.array(direction, dtype=np.float)
        # assert np.issubdtype(type(destination[0]), np.integer)
        # assert np.issubdtype(type(destination[1]), np.integer)

        self.path_v2(direction, speed, mountain_inner_range, mountain_outer_range)


    def path_v3(self, direction, speed, mountain_inner_range=140, mountain_outer_range=160):
        """
        Determine one step that the helicopter or search party will take to reach its destination

        Args:
            direction (_type_): np.array([])
            speed (_type_): _description_
            mountain_inner_range (int, optional): _description_. Defaults to 140.
            mountain_outer_range (int, optional): _description_. Defaults to 160.

        Returns:
            _type_: _description_
        """
        # old_location = self.location.copy()
        # new_location = (np.array([direction[0], direction[1]]) * speed).astype(np.int) + self.location

        # if distance_movement <= speed:
        #     # If the travel speed would overshoot traveling to the intended
        #     # destination, just place helicopter on the destination
        #     new_location = destination
        # else:
        speed = min(speed, self.speed) 
        step = direction * speed
        new_location = np.round([self.location[0] + step[0], self.location[1] + step[1]]).astype(np.int)

        desired_heading = np.arctan2(direction[1], direction[0])
        mountain_dist, mountain_in_range = self.in_range_of_mountain(new_location, mountain_outer_range)
        if mountain_in_range:
            # if direction_mode:
            #     current_location = np.array(new_location.copy())
            #     every_step = (direction * speed).astype(np.int)
            #     while self.in_range_of_mountain(current_location, mountain_outer_range)[1]:
            #         current_location += every_step
                
            #     if self.debug:
            #         print("Adding a location command due to mountain interference of direction command")
            #     self.planned_path.insert(0, ('l', int(current_location[0]), int(current_location[1])))
            #     return False

            # if we are within the inner bounds of the mountain, move directly outwards
            if mountain_dist <= mountain_inner_range:
                theta = self.calculate_desired_heading(self.location, mountain_in_range)
                # theta += np.random.randn() * 0.1  # this could help break infinite oscillation between the agent and the mountain
                if theta < 0:
                    theta += np.pi
                else:
                    theta -= np.pi
            else:
                theta = self.get_angle_away(mountain_in_range, self.location, desired_heading)
                # theta += np.random.randn() * 0.1  # this could help break infinite oscillation between the agent and the mountain
                if theta < -np.pi:
                    theta += 2 * np.pi
                elif theta > np.pi:
                    theta -= 2 * np.pi
            direction = np.array([np.cos(theta), np.sin(theta)])
            step = direction * speed
            self.location = [self.location[0] + int(np.round(step[0])), self.location[1] + int(np.round(step[1]))]
        else:
            self.location = new_location

        if self.terrain.violate_edge_constraints(self.location[0], self.location[1], 1, 1):
            self.location = np.clip(self.location, [0, 0], [self.terrain.dim_x - 1, self.terrain.dim_y - 1]).astype(np.int)
        if isinstance(self.location, np.ndarray):
            self.location = list(self.location)
        assert np.issubdtype(type(self.location[0]), np.integer)
        assert np.issubdtype(type(self.location[1]), np.integer)
        return True

    def plan_spiral(self, ignore_mountain=True):
        """ If ignore mountain, we will not plan a spiral"""
        if ignore_mountain and self.in_range_of_mountain(self.location, 170)[1] is None:
            # don't plan a spiral
            self.plan_path_to_random()

        detection_range = 22  # 100% PoD of SP when in wood and fugitive is fast walking (speed=7.5)
        spiral_unit = 2 * detection_range
        directions_of_spiral = np.array([
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1]
        ])
        current_loc = self.location
        index_of_direction = 0
        current_spiral_coefficient = 1
        while current_spiral_coefficient < 8:
            next_loc = current_loc + \
                         directions_of_spiral[index_of_direction, :] * spiral_unit * current_spiral_coefficient
            if self.terrain.violate_edge_constraints(next_loc[0], next_loc[1], 1, 1):
                break
            self.planned_path.append(('ls', next_loc[0], next_loc[1]))
            index_of_direction = (index_of_direction + 1) % 4
            if index_of_direction == 0 or index_of_direction == 2:
                current_spiral_coefficient += 1
            current_loc = next_loc
        if len(self.planned_path) == 0:
            self.plan_path_to_random()

    def plan_path_to_loc(self, loc):
        not_mountain_loc = loc.copy()
        sample_radius = 200
        while (self.terrain.violate_edge_constraints(not_mountain_loc[0], not_mountain_loc[1], 1, 1)) or \
                self.in_range_of_mountain(not_mountain_loc, 161)[1]:
            not_mountain_loc = np.array([np.random.randint(loc[0] - sample_radius, loc[0] + sample_radius),
                                         np.random.randint(loc[1] - sample_radius, loc[1] + sample_radius)],
                                        dtype=np.int)
        self.planned_path = [('l', not_mountain_loc[0], not_mountain_loc[1])]

    def plan_path_to_random(self):
        sampled_destination = self.sample_destinations(sample_method='random')
        self.plan_path_to_loc(sampled_destination)

    def plan_path_to_intercept(self, speed, direction, current_loc):
        """
        Determine the path the helicopter or search party will take to intercept the fugitive
        Note this planning does not account for mountains
            :param speed: the speed of the fugitive calculated
            :param direction: the direction towards which the fugitive is moving  (the angle is [-pi, pi])
            :param current_loc: the location of the fugitive right now
        """
        # if the fugitive is within the range of one step, just do it and no need for interception
        path_vector = np.array([current_loc[0] - self.location[0], current_loc[1] - self.location[1]])
        distance_movement = np.sqrt(path_vector[0] ** 2 + path_vector[1] ** 2)
        if distance_movement < self.speed:
            self.planned_path = [('l', current_loc[0], current_loc[1])]
            return

        # calculate the line of the fugitive movement, y=k*x+b
        k_fugitive = np.tan(direction)
        b_fugitive = current_loc[1] - k_fugitive * current_loc[0]

        if np.abs(direction) < 1e-3:
            # the fugitive moves horizontally, need special treatment since the slope calculation will be buggy
            x_tangent = self.location[0]
            y_tangent = current_loc[1]
        else:
            # calculate the blue movement if going directly to the tangent point
            k_blue = -1 / k_fugitive
            b_blue = self.location[1] - k_blue * self.location[0]

            # calculate the tangent point
            x_tangent = (b_blue - b_fugitive) / (k_fugitive - k_blue)
            y_tangent = k_blue * x_tangent + b_blue
            x_tangent, y_tangent = np.round(x_tangent).astype(np.int), np.round(y_tangent).astype(np.int)
        tangent_point = np.array([x_tangent, y_tangent], dtype=np.float)

        # direction of the tangent point from the current fugitive loc
        direction_tangent = np.array([x_tangent - current_loc[0], y_tangent - current_loc[1]], dtype=np.float)
        is_blue_on_fugitive_movement_side = (direction_tangent[0] * np.cos(direction) > 0)
        # print(self, "is_blue_on_fugitive_side", is_blue_on_fugitive_movement_side)
        if is_blue_on_fugitive_movement_side:
            if self.terrain.violate_edge_constraints(x_tangent, y_tangent, 1, 1):
                # tangent point is outside the terrain, go to the nearest point to the tangent point in terrain
                x_tangent = np.clip(x_tangent, 0, self.terrain.dim_x - 1).astype(np.int)
                y_tangent = np.clip(y_tangent, 0, self.terrain.dim_y - 1).astype(np.int)
                self.planned_path = [('l', x_tangent, y_tangent)]
            else:
                time_to_tangent_fugitive = distance(current_loc, tangent_point) / speed
                time_to_tangent_blue = distance(self.location, tangent_point) / self.speed
                if time_to_tangent_fugitive < time_to_tangent_blue:
                    # fugitive arrives at the tangent point earlier than the blue
                    # go to the tangent point and follow the fugitive's direction
                    self.plan_path_to_loc([x_tangent, y_tangent])
                    # the distance between blue and red when the blue arrives the tangent point
                    distance_to_go = speed * time_to_tangent_blue - distance(current_loc, tangent_point)
                    buffer_distance = 30
                    time_to_buffer = (distance_to_go - buffer_distance) / (self.speed - speed)
                    if time_to_buffer > 1:
                        # could have a few steps of fast movements
                        self.planned_path.append(('d', np.cos(direction), np.sin(direction), time_to_buffer - 1, 'c'))
                    else:
                        # just use constrained speed
                        self.planned_path.append(('d', np.cos(direction), np.sin(direction), -1, 'c'))
                else:
                    # fugitive arrives at the tangent point later than the blue
                    # go to the tangent point and follow the opposite of fugitive's direction
                    self.plan_path_to_loc([x_tangent, y_tangent])
                    # the distance between blue and red when the blue arrives the tangent point
                    distance_to_go = distance(current_loc, tangent_point) - speed * time_to_tangent_blue
                    buffer_distance = 200
                    time_to_buffer = (distance_to_go - buffer_distance) / (self.speed + speed)
                    if time_to_buffer > 1:
                        # could have a few steps of fast movements
                        self.planned_path.append(('d', -np.cos(direction), -np.sin(direction), time_to_buffer - 1, 'm'))
                    else:
                        # just use constrained speed
                        self.planned_path.append(('d', -np.cos(direction), -np.sin(direction), -1, 'm'))
                if time_to_tangent_blue == 0:
                    self.planned_path.pop(0)
        else:
            self.planned_path = [('l', current_loc[0], current_loc[1])]
            # the distance between blue and red when the blue arrives the current point
            distance_to_go = speed * distance(self.location, current_loc) / self.speed
            buffer_distance = 30
            time_to_buffer = (distance_to_go - buffer_distance) / (self.speed - speed)
            if time_to_buffer > 1:
                # could have a few steps of fast movements
                self.planned_path.append(('d', np.cos(direction), np.sin(direction), time_to_buffer - 1, 'c'))
            else:
                # just use constrained speed
                self.planned_path.append(('d', np.cos(direction), np.sin(direction), -1, 'c'))

        # calculate contact-point method (may not be robust, may need debug)
        # if is_blue_on_fugitive_movement_side or self.speed > speed:
        #     # the blue is on the same side of the fugitive movement or
        #     # if the blue's max speed is faster than the fugitive, then the blue can chase the fugitive
        #
        #     # solve for the contact point by polynomial equation
        #     poly_square_coefficient = (speed ** 2 - self.speed ** 2) * (1 + k_fugitive ** 2)
        #     poly_linear_coefficient = 2 * speed ** 2 * (-self.location[0] + k_fugitive * (b_fugitive - self.location[1])) + 2 * self.speed ** 2 * (current_loc[0] - 2 * k_fugitive * (b_fugitive - current_loc[1]))
        #     poly_constant_coefficient = speed ** 2 * (current_loc[0] ** 2 + (b_fugitive - self.location[1]) ** 2) - self.speed ** 2 * (current_loc[0] ** 2 + (b_fugitive - current_loc[1]) ** 2)
        #     equation = np.poly1d(poly_square_coefficient, poly_linear_coefficient, poly_constant_coefficient)
        #     first_sol = equation.r[0]
        #     second_sol = equation.r[1]
        #     assert np.isrealobj(first_sol), "Should exist contact point"
        #     if is_blue_on_fugitive_movement_side:
        #         time_to_tangent_fugitive = AbstractObject.distance(current_loc, [x_tangent, y_tangent]) / speed
        #         time_to_tangent_blue = AbstractObject.distance(self.location, [x_tangent, y_tangent]) / speed
        #         if time_to_tangent_fugitive < time_to_tangent_blue:
        #             # the contact point will be after the tangent point
        #             # only one solution will be on the same side as fugitive movement
        #             if (first_sol[0] - current_loc[0]) * np.cos(direction) + (first_sol[1] - current_loc[1]) * np.sin(direction) < 0:
        #                 first_sol, second_sol = second_sol, first_sol
        #             assert (first_sol[0] - current_loc[0]) * np.cos(direction) + (first_sol[1] - current_loc[1]) * np.sin(direction) > 0, "there has to be a solution on the same side of fugitive movement"
        #             if self.terrain.violate_edge_constraints(first_sol[0], first_sol[1], 1, 1):
        #                 # the contact point is off terrain
        #                 if self.terrain.violate_edge_constraints(x_tangent, y_tangent, 1, 1):
        #                     # go to the tangent point if possible and follow the fugitive's direction
        #                     self.planned_path = [('l', x_tangent, y_tangent),
        #                                          ('d', np.cos(direction), np.sin(direction))]
        #                 else:
        #                     self.planned_path = [('l', current_loc[0], current_loc[1]),
        #                                          ('d', np.cos(direction), np.sin(direction))]
        #             else:
        #                 self.planned_path = [('l', first_sol[0], first_sol[1])]
        #         else:
        #             # the contact point will be before the tangent point
        #             # only one solution will be in between the current fugitive location and the tangent point
        #             if (first_sol[0] - current_loc[0]) * (x_tangent - current_loc[0]) + \
        #                     (first_sol[1] - current_loc[1]) * (y_tangent - current_loc[1]) < 0:
        #                 first_sol, second_sol = second_sol, first_sol
        #             assert (first_sol[0] - current_loc[0]) * (x_tangent - current_loc[0]) + \
        #                    (first_sol[1] - current_loc[1]) * (y_tangent - current_loc[1]) < 0, "there has to be a solution in between the current fugitive location and the tangent point"
        #             if self.terrain.violate_edge_constraints(first_sol[0], first_sol[1], 1, 1):
        #                 # the contact point is off terrain
        #                 # go to the tangent point and follow the fugitive's direction
        #                 self.planned_path = [('l', x_tangent, y_tangent),
        #                                      ('d', np.cos(direction), np.sin(direction))]
        #             else:
        #                 self.planned_path = [('l', first_sol[0], first_sol[1])]
        #     else:
        #         pass
        # else:
        #     # if the blue's max speed is slower than the fugitive, then it is impossible to catch the fugitive
        #     # then we move to the current_loc of the fugitive and then move with the direction
        #     self.planned_path = [('l', current_loc[0], current_loc[1]),
        #                          ('d', np.cos(direction), np.sin(direction))]

    def path(self, camera_list=[], camera_avoid_bool=False, mountain_inner_range = 140, mountain_outer_range = 150):
        """
        Determine the path the helicopter or search party will take to reach its destination
            :param camera_list: a list of camera objects
            :param camera_avoid_bool: set to True to path around cameras
            :param mountain_inner_range: the range at which the agent will move directly away from the mountain center
            :param mountain_outer_range: the range at which the agent will move perpendicular to the mountain
        """
        path_vector = [self.destination[0] - self.location[0],self.destination[1] - self.location[1]]
        distance = np.sqrt(path_vector[0]**2 + path_vector[1]**2)
        if (distance == 0):
            direction = np.zeros_like(path_vector)
        else:
            direction = path_vector / distance
        
        camera_in_range = self.in_range_of_camera(camera_list)
        old_location = self.location.copy()

        if (distance < self.speed):
            # If the travel speed would overshoot traveling to the intended 
            # destination, just place helicopter on the destination
            self.location = self.destination
        else:
            step = direction * self.speed
            new_location = np.array([self.location[0]+int(np.round(step[0])),self.location[1]+int(np.round(step[1]))])
            desired_heading = np.arctan2(direction[1],direction[0])
            mountain_dist, mountain_in_range = self.in_range_of_mountain(new_location, mountain_outer_range)
            
            if mountain_in_range:
                # if we are within the inner bounds of the mountain, move directly outwards
                if mountain_dist <= mountain_inner_range:
                    theta = self.calculate_desired_heading(self.location, mountain_in_range)
                    if theta < 0:
                        theta += np.pi
                    else:
                        theta -= np.pi
                else:
                    theta = self.get_angle_away(mountain_in_range, self.location, desired_heading)
                direction = np.array([np.cos(theta),np.sin(theta)])
                step = direction * self.speed
                self.location = [self.location[0]+int(np.round(step[0])),self.location[1]+int(np.round(step[1]))]
            elif camera_in_range and camera_avoid_bool:
                theta = self.get_angles_away_from_object_location(camera_in_range.location,self.location)[0]
                direction = np.array([np.cos(theta),np.sin(theta)])
                step = direction * self.speed
                self.location = [self.location[0]+int(np.round(step[0])),self.location[1]+int(np.round(step[1]))]
            else:
                step = direction * self.speed
                self.location = np.array([self.location[0]+int(np.round(step[0])),self.location[1]+int(np.round(step[1]))])
        if self.terrain.violate_edge_constraints(self.location[0],self.location[1],1,1):
            self.location = old_location
            self.destination = [np.random.randint(0, 2428),np.random.randint(0, 2428)]

    def get_angle_away(self, object_location, location, theta):
        """ If the object's desired heading is towards the mountain, move at the incident angle away from the mountain. 
            If the object's desired heading is away from the mountain, move directly towards the target. 
            Because our angles wrap around at -pi and pi, the function calculates differently for the third and fourth quadrants
            where adding pi/2 or subtracting pi/2 will push it over the -pi and pi bounds.

            :param object_location: the location of the object we want to avoid
            :param location: our current location
            :param theta: the desired heading of the agent (where we want to go)
            
            """
        location_to_object_theta = np.arctan2(object_location[1] - location[1], object_location[0] - location[0])
        
        # If agent is in 3rd quadrant
        if -np.pi < location_to_object_theta < -np.pi / 2:
            theta_one = location_to_object_theta + np.pi / 2
            theta_two = location_to_object_theta + 3 * np.pi / 2
            if theta < theta_one or theta > theta_two:
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2*np.pi - theta_one), np.abs(theta - 2*np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2*np.pi - theta_two), np.abs(theta - 2*np.pi - theta_two))
                if theta_dist_one < theta_dist_two:
                    return theta_one
                else:
                    return theta_two
            else:
                return theta
        # If agent is in 4th quadrant
        elif np.pi / 2 < location_to_object_theta < np.pi:
            theta_one = location_to_object_theta - np.pi / 2
            theta_two = location_to_object_theta - 3 * np.pi / 2
            if theta > theta_one or theta < theta_two:
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2*np.pi - theta_one), np.abs(theta - 2*np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2*np.pi - theta_two), np.abs(theta - 2*np.pi - theta_two))
                if theta_dist_one < theta_dist_two:
                    return theta_one
                else:
                    return theta_two
            else:
                return theta
        # If agent is in 1st or 2nd quadrant
        else:
            theta_one = location_to_object_theta - np.pi / 2
            theta_two = location_to_object_theta + np.pi / 2
            if theta_one < theta < theta_two:
                theta_dist_one = min(np.abs(theta - theta_one), np.abs(theta + 2*np.pi - theta_one), np.abs(theta - 2*np.pi - theta_one))
                theta_dist_two = min(np.abs(theta - theta_two), np.abs(theta + 2*np.pi - theta_two), np.abs(theta - 2*np.pi - theta_two))
                if theta_dist_one < theta_dist_two:
                    return theta_one
                else:
                    return theta_two
            else:
                return theta

    def check_collision(self, location):
        return self.terrain.world_representation[0, location[0], location[1]] == 1

    def in_range_of_mountain(self, location, range_distance):
        """ Returns the mountain object if within a certain range of the the agent's location

        :param location: the location of the agent
        :param range_distance: the range between the location and the mountain where we will return the mountain location

         """
        for mountain_location in self.terrain.mountain_locations:
            mountain_location_corrected = (mountain_location[1], mountain_location[0])
            # need distance from edge of mountain to center, 120?
            dist = distance(location, mountain_location_corrected)
            if dist <= range_distance:
                return dist, mountain_location_corrected
        return None, None
    
    def in_range_of_camera(self, camera_list):
        """ Return the camera object if if object is within range of the camera """
        buffer_range = 5
        fugitive_speed_floor = 1
        camera_detection_object_type_coefficient = 1.0
        # cameras can detect an object within 4 grids moving with speed 1 with 100% PoD in wood
        # return 4 * self.detection_terrain_coefficient[self.terrain.terrain_given_location(self.location)] * self.detection_object_type_coefficient * speed
        camera_base_100_pod_distance_floor = 4 * camera_detection_object_type_coefficient * fugitive_speed_floor
        max_pod_distance = camera_base_100_pod_distance_floor * 3 + buffer_range
        for camera in camera_list:
            if distance(camera.location, self.location) <= max_pod_distance:
                return camera
        return False

    def calculate_desired_heading(self, start_location, end_location):
        return np.arctan2(end_location[1] - start_location[1], end_location[0] - start_location[0])


class DetectionObject(AbstractObject):
    detection_factor = 4.0
    def __init__(self, terrain, location,
                 detection_object_type_coefficient):
        """
        DetectionObject defines object that is able to detect.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        :param detection_object_type_coefficient: a multiplier of detection due to detection device type (for example, camera = 1, helicopter = 0.5, search party = 0.75)
        """
        # self.detection_terrain_coefficient = {
        #     TerrainType.MOUNTAIN: 1.0,
        #     TerrainType.WOODS: 1.0,
        #     TerrainType.DENSE_FOREST: 0.5
        # }
        self.detection_object_type_coefficient = detection_object_type_coefficient
        AbstractObject.__init__(self, terrain, location)

    def detect(self, location_object, speed_object):
        """
        Determine detection of an object based on its location and speed
        :param location_object:
        :param speed_object:
        :return: [b,x,y] where b is a boolean indicating detection, and [x,y] is the location of the object in world coordinates if b=True, [x,y]=[-1,-1] if b=False
        """
        distance = np.sqrt(np.square(self.location[0]-location_object[0]) + np.square(self.location[1]-location_object[1]))
        base_100_pod_distance = self.base_100_pod_distance(speed_object)
        if distance < base_100_pod_distance:  # the PoD is 100% within base_100_pod_distance
            return [1, location_object[0]/2428, location_object[1]/2428]
        if distance > base_100_pod_distance * 3:  # the PoD is 0% outside 3*base_100_pod_distance
            return [0, -1, -1]
        probability_of_detection = 1 - (distance - base_100_pod_distance) / (2 * base_100_pod_distance)  # the PoD is linear within [base_100_pod_distance, 3*base_100_pod_distance]
        if np.random.rand() < probability_of_detection:
            return [1, location_object[0]/2428, location_object[1]/2428]
        else:
            return [0, -1, -1]

    @property
    def detection_range(self):
        """
        get detection range of PoD > 0 assuming the object is fast walk speed
        :return: the largest detection range of PoD > 0
        """
        return self.base_100_pod_distance(speed=7.5) * 3

    def base_100_pod_distance(self, speed):
        """
        Calculate the distance within which the Probability of Detection is 100%
        :param speed: the speed of the detected object
        :return: the maximum distance of 100% PoD
        """
        # cameras can detect an object within 4 grids moving with speed 1 with 100% PoD in wood
        return self.detection_factor * self.terrain.detection_coefficient_given_location(self.location) * self.detection_object_type_coefficient * speed
