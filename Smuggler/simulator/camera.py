from simulator.abstract_object import AbstractObject, DetectionObject
from simulator.terrain import TerrainType


class Camera(DetectionObject):
    def __init__(self, terrain, location, known_to_fugitive, detection_object_type_coefficient=1.0):
        """
        Camera defines camera objects. Initializes detection parameters.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        :param known_to_fugitive: boolean denoting whether the camera is known to the fugitive
        """
        DetectionObject.__init__(self, terrain, location, detection_object_type_coefficient)
        self.known_to_fugitive = known_to_fugitive
