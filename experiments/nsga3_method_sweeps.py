"""This module contains the code for the confidence based resampling (CBR) NSGA-III experiments with all three methods."""

# Make other project modules accessible
import sys
import os

sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logging config import
from config.logging_config import setup_logging

# Import Problems, Algorithms and Callback
from src.problems.noisy_dtlz import *
from src.problems.noisy_wfg import *
from src.problems.noisy_zdt import *
from src.algorithms.resampling_adaptations.cbr_nsga3 import CBR_NSGA3
from src.callbacks.cbr_callback import CBR_Callback

# Package imports
import numpy as np

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

# Python imports
import random
from pathlib import Path
import argparse
import multiprocessing as mp

# Results path as global variable
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

def method_sweep(seed: int, threshold: float, num_neighbors: int, method: str, problem: str, std: float, noise_type: str):
    """Method that starts the experiments with different resampling methods.

    Args:
        seed (int): Experiment seed for RNG.
        threshold (float): Threshold that should be tested.
        num_neighbors (int): Number of neighbors used for UCB formula.
        method (str): Resampling method ("neighbors", "population", or "generation").
        problem (Problem): Pymoo problem that is to be optimized.
    """

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)

    # Initialize Algorithm, Callback and Problem
    algo_instance = CBR_NSGA3(resampling_threshold=threshold, num_neighbors=num_neighbors, method=method)
    problem_instance = None

    match problem:
        case "NZDT1": problem_instance = NZDT1(noise_std=std, noise_type=noise_type)
        case "NZDT2": problem_instance = NZDT2(noise_std=std, noise_type=noise_type)
        case "NZDT3": problem_instance = NZDT3(noise_std=std, noise_type=noise_type)
        case "NZDT4": problem_instance = NZDT4(noise_std=std, noise_type=noise_type)
        case "NZDT5": problem_instance = NZDT5(noise_std=std, noise_type=noise_type)
        case "NZDT6": problem_instance = NZDT6(noise_std=std, noise_type=noise_type)
        case "NDTLZ1": problem_instance = NDTLZ1(noise_std=std, noise_type=noise_type)
        case "NDTLZ2": problem_instance = NDTLZ2(noise_std=std, noise_type=noise_type)
        case "NDTLZ3": problem_instance = NDTLZ3(noise_std=std, noise_type=noise_type)
        case "NDTLZ7": problem_instance = NDTLZ7(noise_std=std, noise_type=noise_type)

    callback_instance = CBR_Callback()

    minimize(
        problem=problem_instance,
        algorithm=algo_instance,
        callback = callback_instance,
        termination=("n_eval", 200000)
    )

    # Save data to pickle files
    param_dir = f"tau_{threshold:.3f}_nn_{num_neighbors}_{method}" if method == "neighbors" \
        else f"tau_{threshold:.3f}_{method}"

    path = _result_dir(
        sweep = "method_sweep",
        algo = type(algo_instance).__name__.lower(),
        problem = problem,
        noise_type = noise_type,
        std = std,
        param_dir = param_dir,
        seed = seed,
    )
    callback_instance.save_data(data_path = path, wipe_old_data=False)

def _result_dir(sweep: str, algo: str, problem: str, noise_type: str, std: float, param_dir: str, seed: int) -> Path:
    """Helper function to create result directory path."""
    return RESULTS_ROOT / sweep / algo / problem / f"{noise_type}_std_{std:.2f}" / param_dir / f"seed_{seed:04d}.pkl"

if __name__ == "__main__":
    #setup_logging()
    parser = argparse.ArgumentParser(
        prog = "Noisy NSGA-III resampling methods sweeps",
        description = "Runs experiments with different CBR resampling methods."
    )
    parser.add_argument("--algo", type=str, required=True, choices=["cbr"],
                   help="cbr = CBR-NSGA-III with different methods")
    parser.add_argument("--problem", type=str, required=True,
                   choices=['NZDT1', 'NZDT2', 'NZDT3', 'NZDT4', 'NZDT6', 'NDTLZ1', 'NDTLZ2', 'NDTLZ3', 'NDTLZ7'])
    parser.add_argument("--std", type=float, required=True)
    parser.add_argument("--noise-type", type=str, required=True,
                   choices=["gaussian", "..."])
    parser.add_argument("--n-seeds", type=int, default=31, help="Seeds 0 .. n-seeds-1. Defaults to 31.")

    # Method-specific parameters
    parser.add_argument("--threshold", type=float, required=True,
                   help="Resampling threshold tau.")
    parser.add_argument("--num-neighbors", type=int, default=5,
                   help="Number of neighbors used for UCB formula. Defaults to 5.")
    parser.add_argument("--method", type=str, required=True,
                   choices=["neighbors", "population", "generation"],
                   help="Resampling method: neighbors, population, or generation.")

    args = parser.parse_args()

    seeds = list(range(args.n_seeds))

    if args.algo == "cbr":
        func = method_sweep
        tasks = [(s, args.threshold, args.num_neighbors, args.method, args.problem, args.std, args.noise_type) for s in seeds]

        with mp.Pool(processes=len(seeds)) as pool:
            pool.starmap(func, tasks)
    else:
        parser.error("Only --algo cbr is supported currently.")
