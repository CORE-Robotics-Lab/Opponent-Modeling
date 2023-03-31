import numpy as np
import matplotlib.pyplot as plt

from simulator.abstract_object import MovingObject, DetectionObject
from simulator.terrain import TerrainType
from .utils import clip_theta, distance, pick_closer_theta
import simulator.fugitive
from simulator.camera import Camera

class Helicopter(MovingObject, DetectionObject):
    def __init__(self, terrain, location, speed):
        """
        Helicopter defines helicopter objects. Initializes detection parameters.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        :param speed: a number representing speed with unit in grids
        """
        MovingObject.__init__(self, terrain, location, speed)
        DetectionObject.__init__(self, terrain, location, detection_object_type_coefficient=2.0)

    @property
    def detection_range(self):
        """
        get detection range of PoD > 0 assuming the object is sprint walk speed
        :return: the largest detection range of PoD > 0
        """
        return self.base_100_pod_distance(speed=15) * 3

    def detect(self, location_object, speed):
        """
        Determine detection of an object based on its location and type of the object
        The helicopters' detection of the fugitive is different than the fugitive's detection of the helicopters as given in the "detection ranges.xlsx".
        :param location_object:
        :param object_instance: the instance referred to the object the fugitive is detecting.
        :return: [b,x,y] where b is a boolean indicating detection, and [x,y] is the location of the object in world coordinates if b=True, [x,y]=[-1,-1] if b=False
        """
        return DetectionObject.detect(self, location_object, speed)

    def path(self, camera_list=[], camera_avoid_bool=False):
        """ Pathing for helicopter, we set the inner and outer mountain ranges to be different than other agents"""
        super().path(camera_list, camera_avoid_bool, 140, 200)