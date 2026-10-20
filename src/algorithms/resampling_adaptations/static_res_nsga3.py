"""This module contains a static resampling version of NSGA-III."""

# Make other project modules accessible
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Package import
import numpy as np

# Pymoo imports
from pymoo.algorithms.moo.nsga3 import NSGA3, ReferenceDirectionSurvival
from pymoo.util.ref_dirs import get_reference_directions

# CB_Resampling import
from src.resampling.cb_resampling import CB_Resampling

# Python imports
import math

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

class SR_NSGA3(NSGA3, LoggingMixin):
    """This class implements NSGA-III for noisy problems with static resampling."""

    def __init__(self, pop_size=100, k = 10, ref_dirs:np.ndarray = None):
        """Init function for static resampling NSGA-III.

        Args:
            pop_size (int, optional): Population size used by NSGA-III. Defaults to 100.
            k (int, optional): Number of re_evaluations per individual. Defaults to 10.
            ref_dirs (np.ndarray, optional): Reference directions. Defaults to None, which
                derives das-dennis directions from the problem's n_obj once it is known."""

        self.k = k
        self._auto_ref_dirs = ref_dirs is None
        super().__init__(ref_dirs = ref_dirs if ref_dirs is not None else _placeholder_ref_dirs(), pop_size=pop_size)
        self.logger.info(f"Instance of {self.__class__.__name__} created. Chosen k = {self.k}.")

    def _setup(self, problem, **kwargs):
        if self._auto_ref_dirs:
            self.ref_dirs = _das_dennis_ref_dirs(problem.n_obj)
            self.survival = ReferenceDirectionSurvival(self.ref_dirs)
        return super()._setup(problem, **kwargs)

    def _initialize_advance(self, infills=None, **kwargs) -> None:
        # CB resampling can be used here to prepare infills
        # Things to keep in mind: It just sets up the variables for infills
        # The new individuals will still need to be sampled with k-1 times.

        # Re-use CBR resampling preparation of infills
        CB_Resampling._prepare_infills(infills = infills)

        # Resampling step (k-1) times
        self.logger.debug("Starting resampling of infills.")
        self._static_resampling(infills = infills)

        return super()._initialize_advance(infills, **kwargs)

    def _advance(self, infills=None, **kwargs) -> None:

        # Prepare new infills with necessary arguments
        CB_Resampling._prepare_infills(infills = infills)

        # Resample
        self._static_resampling(infills = infills)

        return super()._advance(infills, **kwargs)

    def _static_resampling(self, infills) -> None:
        """Method that implements static resampling for individuals."""
        if infills is not None:
            for ind in infills:
                while ind.n_evals < self.k:
                    self.evaluator.eval(problem = self.problem, pop = ind, algorithm = self, skip_already_evaluated=False)
                    ind.old_mean = ind.mean
                    ind.n_evals += 1
                    ind.f_sum += ind.F
                    ind.mean = ind.f_sum / ind.n_evals
                    ind.F = ind.mean
