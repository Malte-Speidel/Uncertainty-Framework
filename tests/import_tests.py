"""This module contains tests that check all top-level project modules can be imported properly."""
# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import importlib

class TestImports(unittest.TestCase):

    def test_algorithms_import(self):
        importlib.import_module("src.algorithms")

    def test_problems_import(self):
        importlib.import_module("src.problems")

    def test_config_import(self):
        importlib.import_module("config")

if __name__ == "__main__":
    unittest.main()
