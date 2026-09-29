"""This module contains tests for the CB_Resampling class."""

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

import unittest

# Import CB_Resampling
from src.resampling.cb_resampling import CB_Resampling

# Package imports
import numpy as np

class TestCbResampling(unittest.TestCase):

    def setUp(self):
        # Minkowski distance test
        self.x = [1,1]
        self.y = [3,3]

    def test_minkowski_euclidean(self):
        euclidean = CB_Resampling._minkowski_dist(self.x, self.y)
        assert euclidean == np.sqrt(8), f"Minkowski euclidean distance should be {np.sqrt(8)}, was {euclidean}"
    
    def test_minkowski_manhattan(self):
        manhattan = CB_Resampling._minkowski_dist(self.x, self.y,p=1)
        assert manhattan == 4, f"Minkowski manhattan distance should be 4, was {euclidean}"

if __name__ == "__main__":
    unittest.main()