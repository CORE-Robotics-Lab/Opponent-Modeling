from simulator.abstract_object import AbstractObject


class Hideout(AbstractObject):
    def __init__(self, terrain, location, known_to_good_guys):
        """
        Hideout defines hideout locations.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        :param known_to_good_guys: boolean denoting whether the camera is known to the good guys
        """
        super(Hideout, self).__init__(terrain, location)
        self.known_to_good_guys = known_to_good_guys
