from collections import defaultdict

import numpy as np

from hyperheuristic.orchestrator.metrics.Metric import Metric


class AggregatedConsistencyMetric(Metric):

    def __init__(self, set_divisor):
        self.set_divisor = set_divisor

    def compute(self, ensemble_particles_history, maximize=True):
        overall_solutions_per_iteration = self.merge_solutions(ensemble_particles_history)
        coefficients = self.compute_performance_coefficients(overall_solutions_per_iteration, maximize)
        return self.compute_mca(ensemble_particles_history, coefficients)

    def compute_mca(self, ensemble_particles_history, coefficients):
        coefficients_per_agent = defaultdict(list)
        for agent_id in range(0, len(ensemble_particles_history)):
            for iteration in ensemble_particles_history[0]:
                coefficients_per_agent[agent_id].append(coefficients[iteration][agent_id])

        avg_mic = 0
        for agent_id in range(0, len(ensemble_particles_history)):
            sd = np.std(coefficients_per_agent[agent_id])
            min_coef = np.min(coefficients_per_agent[agent_id])
            max_coef = np.max(coefficients_per_agent[agent_id])
            avg_coef = np.average(coefficients_per_agent[agent_id])
            max_sd = np.sqrt(((min_coef - avg_coef) ** 2 + (max_coef - avg_coef) ** 2) / 2)
            max_sd = 1 if max_sd == 0 else max_sd
            mic = sd / max_sd
            avg_mic += mic / len(ensemble_particles_history)
        mca = 1 - avg_mic
        return mca

    def merge_solutions(self, ensemble_particles_history):
        overall_solutions_per_iteration = {}
        for iteration in ensemble_particles_history[0]:
            merged_solutions = []
            for idx, val in enumerate(ensemble_particles_history):
                particles = val[iteration]
                merged_solutions = np.append(merged_solutions, particles)
                overall_solutions_per_iteration[iteration] = merged_solutions
        return overall_solutions_per_iteration

    def compute_performance_coefficients(self, overall_solutions_per_iteration, maximize=True):
        coefficients = {}
        for iteration in overall_solutions_per_iteration:
            sorted_solutions = sorted(overall_solutions_per_iteration[iteration], key=lambda p: p.value,
                                      reverse=maximize)[
                               :len(overall_solutions_per_iteration[iteration]) // self.set_divisor]
            coefficients[iteration] = defaultdict(int)
            for p in sorted_solutions:
                coefficients[iteration][p.id] += 1 / (len(sorted_solutions))
        return coefficients
