from Prison_Escape.environment.abstract_object import MovingObject, DetectionObject
from Prison_Escape.environment.terrain import TerrainType
import Prison_Escape.environment.helicopter
import Prison_Escape.environment.search_party


class Fugitive(DetectionObject):
    def __init__(self, terrain, location):
        """
        Fugitive defines the fugitive. Initializes detection parameters.
        :param terrain: a terrain instance
        :param location: a list of length 2. For example, [5, 7]
        """
        DetectionObject.__init__(self, terrain, location, detection_object_type_coefficient=0.5)
        # NOTE: the detection_object_type_coefficient is variant for fugitive as it is detecting different objects

    def detect(self, location_object, object_instance):
        """
        Determine detection of an object based on its location and type of the object
        The fugitive's detection of other parties is different other parties' detection of the fugitive as given in the "detection ranges.xlsx".
        The fugitive's detection of other parties depends on what the party is.
        :param location_object:
        :param object_instance: the instance referred to the object the fugitive is detecting.
        :return: [b,x,y] where b is a boolean indicating detection, and [x,y] is the location of the object in world coordinates if b=True, [x,y]=[-1,-1] if b=False
        """
        if isinstance(object_instance, Prison_Escape.environment.helicopter.Helicopter):
            self.detection_object_type_coefficient = 0.5
            return DetectionObject.detect(self, location_object, 8)
        elif isinstance(object_instance, Prison_Escape.environment.search_party.SearchParty):
            self.detection_object_type_coefficient = 0.75
            return DetectionObject.detect(self, location_object, 3)
        else:
            raise NotImplementedError
