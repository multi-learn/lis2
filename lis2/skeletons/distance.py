import abc
import math

from configurable import TypedConfigurable, Schema


class BaseDistance(TypedConfigurable):
    """
    BaseDistance for all distances algorithms.

    Configuration :
        - **name** (str): The name of distance, BaseDistance, used.
    """

    @abc.abstractmethod
    def get_distance(self, p1, p2):
        """
        Obtain distance between two points.

        Args:
            p1 (np.array) : Point of space containing x,y,z coordinates along velocity and density at these coordinates.
            p2 (np.array) : Point of space containing x,y,z coordinates along velocity and density at these coordinates.

        Returns:
            The distance between p1 and p2
        """
        pass


class DistanceEuclieanExp(BaseDistance):
    """
    Computes a modified Euclidean distance that increases exponentially when the speed difference
    between two points exceeds a specified threshold.

    This distance metric calculates the standard Euclidean distance in 3D space and applies an
    exponential penalty if the absolute speed difference between the two points surpasses
    `speed_threshold`. This ensures that significant speed variations contribute more heavily to
    the computed distance.

    Configuration :
        - **speed_threshold** (float): threshold of speed difference.
    """

    config_schema = {
        "speed_threshold": Schema(float, default=10.0),
    }

    def get_distance(self, p1, p2):
        x1, y1, z1, v1, _ = p1
        x2, y2, z2, v2, _ = p2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) + math.exp(
            abs(v1 - v2) - self.speed_threshold
        )


class DistanceSpeed(BaseDistance):
    """
    Computes the distance metric as the absolute difference between the speeds of two points.

    This metric ignores spatial positioning and focuses solely on the difference in velocity.
    """

    def get_distance(self, p1, p2):
        _, _, _, v1, _ = p1
        _, _, _, v2, _ = p2
        return abs(v1 - v2)
