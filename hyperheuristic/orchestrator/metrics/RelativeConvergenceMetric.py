from collections import defaultdict
import numpy as np

from hyperheuristic.orchestrator.metrics.Metric import Metric


class RelativeConvergenceMetric(Metric):

    def __init__(self, set_divisor):
        self.set_divisor = set_divisor

    def compute(self, ensemble_particles_history, maximize=True):
        overall_solutions_per_iteration = self.merge_solutions(ensemble_particles_history)
        return self.compute_performance_coefficients(overall_solutions_per_iteration, maximize)

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
        coefficients = defaultdict(int)
        for iteration in overall_solutions_per_iteration:
            sorted_solutions = sorted(overall_solutions_per_iteration[iteration], key=lambda p: p.value,
                                      reverse=maximize)[
                               :len(overall_solutions_per_iteration[iteration]) // self.set_divisor]
            for p in sorted_solutions:
                coefficients[p.id] += 1 / (len(sorted_solutions) * len(overall_solutions_per_iteration))
        return coefficients

    def compute_performance_coefficients_2(self, overall_solutions_per_iteration, maximize=True):
        coefficients = defaultdict(int)
        for iteration in overall_solutions_per_iteration:
            sorted_solutions = sorted(overall_solutions_per_iteration[iteration], key=lambda p: p.value,
                                      reverse=maximize)[
                               :2 * len(overall_solutions_per_iteration[iteration]) // self.set_divisor]
            for p in sorted_solutions:
                coefficients[p.id] += 1  # /(len(sorted_particles))
            max_id = max(coefficients, key=coefficients.get)
            sum = 0
            for id in coefficients:
                coefficients[id] = coefficients[id] / coefficients[max_id]
                sum += coefficients[id]
        return coefficients

    def compute_performance_coefficients_3(self, overall_particles_per_iteration, maximize=True):
        coefficients = defaultdict(int)
        for iteration in overall_particles_per_iteration:
            for p in overall_particles_per_iteration[iteration]:
                if maximize and coefficients.get(p.id, -100000000000) < p.value:
                    coefficients[p.id] = p.value
                elif not maximize and coefficients.get(p.id, 100000000000) > p.value:
                    coefficients[p.id] = p.value
            max_id = max(coefficients, key=coefficients.get)
            max_value = coefficients[max_id]
            sum = 0
            for id in coefficients:
                # coefficients[id] = coefficients[id]/(max_value*len(overall_particles_per_iteration[iteration]) * len(overall_particles_per_iteration))
                sum += coefficients[id] / max_value
            for id in coefficients:
                # coefficients[id] = coefficients[id]/(max_value*len(overall_particles_per_iteration[iteration]) * len(overall_particles_per_iteration))
                coefficients[id] = coefficients[id] / (max_value * sum * len(overall_particles_per_iteration))
        return coefficients
