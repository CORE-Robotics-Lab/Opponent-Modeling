from simulator.abstract_object import AbstractObject


class Town(AbstractObject):
    def __init__(self, terrain, location):
        """
        Town defines towns.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        """
        super(Town, self).__init__(terrain, location)
