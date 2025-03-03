import abc

import skimage.morphology as morph
from fil_finder import FilFinder2D
import astropy.units as u

from configurable import TypedConfigurable, Schema

class BaseSkeletonize(abc.ABC, TypedConfigurable):
    """
    BaseSkeletonize for all skeletonizations algorithms.

    This class serves as a template for extracting individual skeletons from masks.
    Subclasses must implement the abstract methods to define specific behavior for these steps.

    Configuration:
        - **name** (str): The name of skeletization algorithm, BaseSkeletonize, used.
    """
    @abc.abstractmethod
    def get_skeletons(self, mask):
        """
        Obtain skeleton from a binary mask.

        Args:
            mask (np.array) : 2d Binary array containing the mask.

        Returns:
            Array containing the skeleton of the mask.
        """
        pass    


class NoSkeleton(BaseSkeletonize):
    """
    NoSkeleton to return the mask without skeletonization.

    Useful to study the full mask. But computational time is longer.
    """
    def get_skeletons(self, mask):
        return mask
    
    
class SkeletonSkimage(BaseSkeletonize):
    """
    SkeletonSkimage to return the skeleton found with skimage.

    The skimage.morphology.skeletonize function thins a binary image 
    to a single-pixel-wide skeleton while preserving connectivity using a morphological thinning algorithm.
    """
    def get_skeletons(self, mask):
        skeletons = morph.skeletonize(mask)
        return skeletons


class SkeletonFilFinder(BaseSkeletonize):
    """
    SkeletonFilFinder2D to return the skeleton with Fil Finder.

    FilFinder 2D is a library for extracting and analyzing filamentary structures 
    in astronomical images using adaptive thresholding, smoothing, and skeletonization to trace filaments and measure their properties.
    The medskel function in FilFinder 2D computes the medial skeleton of a binary image while preserving topology, similar to a constrained medial axis transform.
    """
    def get_skeletons(self, mask):
        fil = FilFinder2D(mask, distance=250 * u.pc, mask=mask)
        fil.medskel()
        skeletons = fil.skeleton
        return skeletons
