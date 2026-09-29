"""This module contains tests for the CBR_NSGA2 class."""
# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import tempfile
import pickle

# Package imports
import numpy as np

# CBR-NSGA2 import
from src.algorithms.resampling_adaptations.cbr_nsga2 import CBR_NSGA2

# CBR Callback import
from src.callbacks.cbr_callback import CBR_Callback

# Problem import
from src.problems.noisy_zdt import NZDT1

# Pymoo imports
from pymoo.optimize import minimize

class TestCbrNsga2(unittest.TestCase):

    def setUp(self):
        self.problem = NZDT1(noise_std=0.6)
        self.algorithm = CBR_NSGA2(resampling_threshold=0.5)
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

if __name__ == "__main__":
    unittest.main()
