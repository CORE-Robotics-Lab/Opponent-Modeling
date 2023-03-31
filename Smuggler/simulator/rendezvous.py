from simulator.abstract_object import AbstractObject


class Rendezvous(AbstractObject):
    def __init__(self, terrain, location):
        """
        Hideout defines hideout locations.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        """
        super(Rendezvous, self).__init__(terrain, location)
        self.reached = False