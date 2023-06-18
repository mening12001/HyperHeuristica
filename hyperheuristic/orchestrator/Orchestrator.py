import abc
import concurrent.futures
import copy
from operator import attrgetter
import random

import numpy as np

from hyperheuristic.orchestrator.metrics.RelativeConvergenceMetric import RelativeConvergenceMetric
from hyperheuristic.orchestrator.metrics.RelativeDiversityMetric import RelativeDiversityMetric


class Orchestrator(metaclass=abc.ABCMeta):
    n_quota_of_solutions = 0
    dimensions = 0
    objective_func = 0
    window_size = 0
    convergence_metric = RelativeConvergenceMetric(set_divisor=3)
    diversity_metric = RelativeDiversityMetric()
    maximize = True
    overall_solutions_state = None
    overall_global_solutions_state = None
    bounds = None
    iteration = 0

    def __init__(self, objective_func, dimensions, bounds, n_quota_of_particles, window_size,
                 maximize=True):
        self.dimensions = dimensions
        self.objective_func = objective_func
        self.n_quota_of_solutions = n_quota_of_particles
        self.window_size = window_size
        self.maximize = maximize
        self.bounds = bounds

    @abc.abstractmethod
    def compose(self, population, tournament_proportion=None):
        pass

    def orchestrate(self, population, tournament_proportion=None):
        ensemble_solutions_history = []
        ensemble_last_solutions = []
        agent_ensemble = self.compose(population, tournament_proportion)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(agent.solve): agent for agent in agent_ensemble}
            for fut in concurrent.futures.as_completed(futures):
                solutions_per_iteration = fut.result()
                ensemble_solutions_history.append(solutions_per_iteration)
                ensemble_last_solutions = np.append(ensemble_last_solutions,
                                                    solutions_per_iteration[self.window_size - 1])

        convergence_coefficients = self.convergence_metric.compute(ensemble_solutions_history, self.maximize)
        #diversity_coefficients = self.diversity_metric.compute(ensemble_solutions_history, self.maximize)

        #for i in range(0, len(convergence_coefficients)):
            #convergence_coefficients[i] = 0.6 * convergence_coefficients[i] + 0.4 * diversity_coefficients[i]


        if self.maximize is True:
            ensemble_global_solution = max(ensemble_last_solutions, key=attrgetter('value'))
        else:
            ensemble_global_solution = min(ensemble_last_solutions, key=attrgetter('value'))

        print("Hypergeneration finished | best cost: {} best position: {}".format(ensemble_global_solution.value,
                                                                                  ensemble_global_solution.position))

        self.update_internal_state(ensemble_last_solutions, ensemble_global_solution)
        return ensemble_last_solutions, convergence_coefficients, ensemble_global_solution

    def update_internal_state(self, ensemble_last_solutions, ensemble_global_solutions):
        self.overall_solutions_state = ensemble_last_solutions
        self.overall_global_solutions_state = ensemble_global_solutions

    def tournament_selection(self, ensemble_solutions, n_quota_of_solutions, tournament_proportion = 0.5):
        if ensemble_solutions is None:
            return None
        temp_ensemble_solutions = copy.deepcopy(ensemble_solutions)
        selected = random.choices(temp_ensemble_solutions, k=int(len(ensemble_solutions) * tournament_proportion))
        selected = sorted(selected, key=lambda p: p.value, reverse=self.maximize)[:n_quota_of_solutions]
        return selected
