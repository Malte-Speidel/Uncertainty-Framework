"""This module contains the confidence bound resampling implementation of NSGA3."""

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# CB_Resampling import
from src.resampling.cb_resampling import CB_Resampling

# Package imports
import numpy as np
import math

# Pymoo imports
from pymoo.algorithms.moo.nsga3 import NSGA3, ReferenceDirectionSurvival
from pymoo.util.ref_dirs import get_reference_directions

def _das_dennis_ref_dirs(n_obj: int, target_points: int = 99) -> np.ndarray:
    """Returns das-dennis reference directions with at least target_points vectors.

    Args:
        n_obj (int): Number of objectives.
        target_points (int, optional): Minimum number of reference directions. Defaults to 99.

    Returns:
        np.ndarray: Reference directions.
    """
    if n_obj == 2:
        return get_reference_directions("das-dennis", n_obj, n_points=target_points)
    p = 1
    while math.comb(p + n_obj - 1, n_obj - 1) < target_points:
        p += 1
    return get_reference_directions("das-dennis", n_obj, n_partitions=p)

def _placeholder_ref_dirs() -> np.ndarray:
    """Returns a throwaway ref_dirs array, replaced once the real problem is known.

    Returns:
        np.ndarray: A single reference direction.
    """
    return np.array([[1.0]])

class CBR_NSGA3(NSGA3, LoggingMixin):
    """This class implements NSGA3 with CB_Resampling."""

    def __init__(
        self,
        pop_size=100,
        resampling_threshold:float = 1.0,
        num_neighbors:int = 5,
        method:str = "neighbors",
        ref_dirs:np.ndarray = None,
        **kwargs,
    ):
        self._auto_ref_dirs = ref_dirs is None
        super().__init__(ref_dirs = ref_dirs if ref_dirs is not None else _placeholder_ref_dirs(), pop_size=pop_size, **kwargs)
        self.resampling_threshold  = resampling_threshold
        self.num_neighbors = num_neighbors
        self.method = method
        self.logger.info(f"Instance of {self.__class__.__name__} created.")

    def _setup(self, problem, **kwargs):
        if self._auto_ref_dirs:
            self.ref_dirs = _das_dennis_ref_dirs(problem.n_obj)
            self.survival = ReferenceDirectionSurvival(self.ref_dirs)
        return super()._setup(problem, **kwargs)

    def _initialize_advance(self, infills=None, **kwargs):
        self.logger.debug("Initializing advance")

        # Prepare infills based on CB_Resampling requirements
        CB_Resampling._prepare_infills(infills = infills)

        return super()._initialize_advance(infills, **kwargs)

    def _advance(self, infills=None, **kwargs):
        self.logger.debug("Advancing ...")

        # Prepare infills based on CB_Resampling requirements
        CB_Resampling._prepare_infills(infills = infills)

        self.logger.debug("Calling threshold resampling")
        CB_Resampling.resample(algo_instance=self, dynamic_threshold=False, threshold=self.resampling_threshold, num_neighbors=self.num_neighbors, method=self.method)

        return super()._advance(infills, **kwargs) # Survival happens in this step