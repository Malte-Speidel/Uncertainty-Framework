"""This module contains the implementation of confidence bound resampling."""

# Make other project modules accessible
import sys
import os
sys.path.insert(1, os.path.join(os.path.dirname(__file__), ".."))

# Logger import
from config.logging_mixin import LoggingMixin

# Package imports
import numpy as np

# Pymoo imports
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population
from pymoo.core.individual import Individual

# Python imports
from random import randrange

class CB_Resampling(LoggingMixin):
    """Class that contains the resampling methods."""

    @staticmethod
    def resample(algo_instance: Algorithm, dynamic_threshold: bool = True, threshold: np.ndarray = None, num_neighbors: int = 5, method: str = "neighbors", **kwargs):
        """Takes an algorithm instance and applies confidence bound resampling to its individuals.

        Args:
            algo_instance (Algorithm): Instance of the pymoo genetic algorithm.
            dynamic_threshold (bool, optional): Decides if fixed or dynamic threshold is used. Defaults to True.
            threshold (np.ndarray, optional): Threshold for which to re-evaluate (ind.uncertainty >= threshold -> re-eval). Defaults to None.
            num_neighbors (int, optional): Number of neighbors used to determine T in UCB formula. Defaults to 5.
            method (str, optional): Method to calculate T in UCB formula. Options: "neighbors" or "population". Defaults to "neighbors".
        """
        CB_Resampling.logger.debug("Starting population resampling.")

        if dynamic_threshold:
            # Calculate threshold
            dyn_threshold = CB_Resampling._calc_threshold(algo_instance = algo_instance, method = "population")
            CB_Resampling._threshold_resampling(algo_instance=algo_instance, threshold = dyn_threshold, num_neighbors=num_neighbors, method=method)
        else:
            # Use fixed threshold
            CB_Resampling._threshold_resampling(algo_instance = algo_instance, threshold = threshold, num_neighbors=num_neighbors, method=method)


    @staticmethod
    def _prepare_infills(infills=None):
        """Adds re_evaluation counter and running mean to the new infills.

        Args:
            infills (Pymoo infills): New individuals added after recombination and mutation. Defaults to None.
        """
        # Preparing new infills
        if infills is not None:
            CB_Resampling.logger.debug("Adding evaluation counter, f_sum, mean and last threshold to new infills.")
            for ind in infills:
                if not hasattr(ind, "n_evals"):
                    ind.n_evals = 1
                    ind.f_sum = np.zeros_like(ind.F)
                    ind.mean = 0
                    ind.old_mean = np.zeros_like(ind.F)
                    ind.suggested_threshold = np.random.random(len(ind.f_sum))
                ind.f_sum += ind.F
                ind.mean = ind.f_sum / ind.n_evals
                ind.F = ind.mean

    @staticmethod
    def _threshold_resampling(algo_instance: Algorithm, threshold: np.ndarray, num_neighbors: int = 5, method: str = "neighbors"):
        """Re-evaluates individuals that are over the specified threshold.

        Args:
            algo_instance (Algorithm): Instance of the pymoo genetic algorithm.
            threshold (np.ndarray): Threshold for which to re-evaluate (ind.uncertainty >= threshold -> re-eval)
            num_neighbors (int, optional): Number of neighbors used to determine T in UCB formula. Defaults to 5.
            method (str, optional): Method to calculate T in UCB formula. Options: "neighbors", "population", or "generation". Defaults to "neighbors".
        """

        CB_Resampling.logger.debug("Applying CB resampling.")
        # Here we need to decide what T is for the CB formula
        # Preliminary T will be made up off the re-eval counts of k-nearest neighbors
        # Subsequently k is also a tunable parameter

        assert algo_instance.pop is not None, f"Population of {algo_instance.__class__.__name__} was None but should not have been."

        if method not in ("neighbors", "population", "generation"):
            method = "neighbors"

        # _knn computes neighbor data for the whole population in one pass, so it's
        # only ever called once per generation here, not once per individual.
        pop_knn = CB_Resampling._knn(pop = algo_instance.pop, k = num_neighbors) if method == "neighbors" else None

        # Calculate T values and the indicator value --> Decide if resampling is necessary
        for i, ind in enumerate(algo_instance.pop):
            match method:
                case "neighbors":
                    T = pop_knn[i][2]  # Use the pre-calculated neighbor sum from _knn
                case "population":
                    T = CB_Resampling._get_total_evals(algo_instance.pop)
                case "generation":
                    T = algo_instance.evaluator.n_eval

            CB_Resampling.logger.debug(f"T value: {T}")
            uncertainty_indicator = np.sqrt(2*np.log(T)/ind.n_evals)

            CB_Resampling.logger.debug(f"Calculated uncertainty indicator: {uncertainty_indicator}")

            # If uncertainty is higher than threshold resample
            if uncertainty_indicator > threshold:
                algo_instance.evaluator.eval(problem=algo_instance.problem, pop = ind, algorithm = algo_instance, skip_already_evaluated=False)

                # Refresh mean and update counter
                # F here is the value that was obtained from the evaluator
                ind.old_mean = ind.mean
                ind.n_evals += 1
                ind.f_sum += ind.F
                ind.mean = ind.f_sum / ind.n_evals
                ind.F = ind.mean
                ind.last_threshold = ind.suggested_threshold

                # Recalculate individual threshold suggestion
                ind.suggested_threshold = (((ind.old_mean - ind.mean)/ind.old_mean)+1)*ind.last_threshold
                CB_Resampling.logger.debug(f"New suggested threshold: {ind.suggested_threshold}")
            else:
                ind.suggested_threshold = ind.suggested_threshold * 0.9 # Lower the threshold by 10%

    @staticmethod
    def _minkowski_dist(x: list, y: list, p: float = 2) -> float:
        """Calculates the Minkowski dist between two points x and y.

        Args:
            x (list): Point x as list
            y (list): Point y as list
            p (float, optional): P value for formula. When p=2 euclidean distance, when p=1 manhattan distance. Defaults to 2.

        Returns:
            float: Minkowski distance between x and y
        """

        #CB_Resampling.logger.debug("Calculating Minkowski distance.")

        # Safety checks
        assert isinstance(x, list) or isinstance(x, np.ndarray), f"Expected x to be list or np.ndarray but got {type(x)}"
        assert isinstance(y, list) or isinstance(y, np.ndarray), f"Expected y to be list or np.ndarray but got {type(y)}"

        assert np.shape(x) == np.shape(y), f"x and y should be of same shape but x={np.shape(x)}, y={np.shape(y)}"

        # If x or y not np.ndarray convert
        x_tmp = np.array(object=x)
        y_tmp = np.array(object=y)

        minkowski = np.sum(np.abs(x_tmp-y_tmp)**p)**(1/p)

        return minkowski

    @staticmethod
    def _knn(pop: Population, k: int = 5) -> list:
        """Returns a list of k nearest neighbors for every individual in the population.

        Distances are computed on objectives normalized by the population's current
        per-objective range (min-max over every `ind.mean` in `pop`), not on raw
        objective values. Without this, the objective with the largest raw scale
        dominates the Euclidean distance -- e.g. DTLZ1/DTLZ3 objectives can be ~1000x
        larger far from the Pareto front than near it, and DTLZ7's last objective
        is ~10-25x larger than the other two at all times.

        Args:
            pop (Population): Pymoo population for which to compute KNN.
            k (int, optional): Number of neighbors. Defaults to 5.

        Returns:
            list: List of individuals and their KNN.
        """

        CB_Resampling.logger.debug(f"Calculating KNN for population of size {len(pop)}.")
        
        # Store the neighbor sums and return them along with individuals
        neighbors_with_sums = []
        
        means = np.array([ind.mean for ind in pop])
        f_min = means.min(axis=0)
        f_range = means.max(axis=0) - f_min
        f_range[f_range == 0] = 1.0  # Flat objective: every individual is equal on it, so it contributes 0 distance either way.
        normed_means = (means - f_min) / f_range

        for i, ind in enumerate(pop):
            distances = []
            for j, ind2 in enumerate(pop):
                if i == j: continue

                distances.append([ind2, CB_Resampling._minkowski_dist(normed_means[i], normed_means[j])])

            # Sort based on distance to ind
            distances.sort(key=lambda x: x[1])

            # Add to neighbors and calculate neighbor sum for this individual
            neighbor_individuals = [d[0] for d in distances[:k]]
            neighbor_sum = sum(neighbor.n_evals for neighbor in neighbor_individuals)
            
            neighbors_with_sums.append([ind, neighbor_individuals, neighbor_sum]) # Store the sum as well
            
        return neighbors_with_sums

    @staticmethod
    def _get_total_evals(pop: Population):
        """Sums the total evaluations of the current population.

        Args:
            pop (Population): Population to sum up

        Returns:
            int: Total number of evaluations of the population
        """

        evaluations = np.sum([ind.n_evals for ind in pop])
        return int(evaluations)


    @staticmethod
    def _calc_threshold(algo_instance: Algorithm, method: str = "population") -> np.ndarray:
        """Calculates the threshold value using the chosen method.

        Args:
            algo_instance (Algorithm): Instance of the current algorithm.
            method (str, optional): Method that decides how threshold is calculated. Available methods population, ... . Defaults to "population".

        Return:
            float: Calculated threshold.
        """

        CB_Resampling.logger.debug(f"Calculating Threshold")

        # Safety checks
        assert method in ["population"], f"The chosen method for threshold calculation is invalid, options are [population, ...]. Chosen was {method}"
        assert algo_instance.pop is not None and len(algo_instance.pop) > 0, f"The supplied algorithm instance did not have a population (was None or had length 0)."
        assert algo_instance.n_gen is not None, f"Generation counter in algorithm instance was None. It should be a positive integer."
        assert algo_instance.problem is not None, f"Problem of the supplied algorithm instance was None."

        # Case for first generation of algorithm where all individuals have no prior threshold
        # In this case threshold is set to 1 globally TODO: Figure out good value for this case
        CB_Resampling.logger.debug(f"Algorithm Generation {algo_instance.n_gen}")
        if algo_instance.n_gen <= 1:
            CB_Resampling.logger.debug(f"First generation so threshold 1 was used.")
            return np.ones(algo_instance.problem.n_obj)*3

        # Additional attribute check
        assert all(hasattr(ind, "suggested_threshold") for ind in algo_instance.pop), f"Not all individuals in the algorithm instance population had the attribute 'last_threshold'."

        CB_Resampling.logger.debug(f"Calculating threshold using method {method}.")

        mean_threshold = np.zeros_like(algo_instance.problem.n_obj)

        if method == "population":
            # Aggregate all old threshold values
            CB_Resampling.logger.debug(f"Number of individuals that contribute to threshold {len(algo_instance.pop)}")
            mean_threshold = np.mean([ind.suggested_threshold for ind in algo_instance.pop], axis = 0)

            CB_Resampling.logger.debug(f"Calculated mean was {mean_threshold} ...")
        return mean_threshold
