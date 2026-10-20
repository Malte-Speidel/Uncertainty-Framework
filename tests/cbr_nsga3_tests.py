"""This module contains tests for the CBR_NSGA3 class."""
# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import tempfile
import pickle

# Package imports
import numpy as np

# CBR-NSGA3 import
from src.algorithms.resampling_adaptations.cbr_nsga3 import CBR_NSGA3

# CBR Callback import
from src.callbacks.cbr_callback import CBR_Callback

# Problem import
from src.problems.noisy_dtlz import NDTLZ1

# Pymoo imports
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

class TestCbrNsga3(unittest.TestCase):

    def setUp(self):
        self.problem = NDTLZ1(noise_std=0.6)
        self.algorithm = CBR_NSGA3(resampling_threshold=0.5)
        self.callback = CBR_Callback()

    def test_minimize_collects_metrics(self):
        result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            termination=("n_eval", 10000),
            callback=self.callback
        )

        assert result.X is not None, "Minimize did not return any decision variables."
        assert len(self.callback.clean_fronts) > 0, "Callback did not record any clean fronts."
        assert np.isfinite(self.callback.igd_plus[-1]) and self.callback.igd_plus[-1] >= 0, f"Final IGD+ should be finite and non-negative, was {self.callback.igd_plus[-1]}."
        assert np.isfinite(self.callback.gd_plus[-1]) and self.callback.gd_plus[-1] >= 0, f"Final GD+ should be finite and non-negative, was {self.callback.gd_plus[-1]}."

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = os.path.join(tmp_dir, "results.pkl")
            self.callback.save_data(data_path=data_path)

            assert os.path.exists(data_path), f"save_data did not create a file at {data_path}."
            with open(data_path, "rb") as f:
                df = pickle.load(f)
            assert "igd_plus" in df.columns, "Saved dataframe is missing the 'igd_plus' column."

    def test_ref_dirs_derived_from_problem(self):
        result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            termination=("n_eval", 1000),
        )

        ref_dirs = result.algorithm.ref_dirs
        assert ref_dirs.shape[1] == self.problem.n_obj, f"ref_dirs were not built for the problem's n_obj. Expected {self.problem.n_obj}, got {ref_dirs.shape[1]}."

    def test_explicit_ref_dirs_are_kept(self):
        ref_dirs = get_reference_directions("das-dennis", self.problem.n_obj, n_partitions=6)
        algorithm = CBR_NSGA3(resampling_threshold=0.5, ref_dirs=ref_dirs)

        result = minimize(
            problem=self.problem,
            algorithm=algorithm,
            termination=("n_eval", 1000),
        )

        assert np.array_equal(result.algorithm.ref_dirs, ref_dirs), "Explicitly supplied ref_dirs were not kept as-is."

if __name__ == "__main__":
    unittest.main()
