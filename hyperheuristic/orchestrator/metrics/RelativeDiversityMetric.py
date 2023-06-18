from collections import defaultdict

import numpy

from hyperheuristic.orchestrator.metrics.Metric import Metric


class RelativeDiversityMetric(Metric):

    def compute(self, ensemble_particles_history, maximize=True):
        overall_standard_deviation_per_iteration = {}
        for iteration in ensemble_particles_history[0]:
            merged_standard_deviation = []
            for idx, val in enumerate(ensemble_particles_history):
                particles = val[iteration]
                standard_deviation = self.compute_standard_deviation(particles)
                merged_standard_deviation.append(standard_deviation)
                overall_standard_deviation_per_iteration[iteration] = merged_standard_deviation
        return self.compute_diversity_coefficients(overall_standard_deviation_per_iteration)

    def compute_standard_deviation(self, particles):
        id = particles[0].id
        var = numpy.std([particle.position for particle in particles], axis=0)
        return {'id' : id, 'sd' : numpy.average(var)}

    def compute_diversity_coefficients(self, overall_particles_per_iteration):
        coefficients = defaultdict(int)
        for iteration in overall_particles_per_iteration:
            sd_iteration_max =  max([particle['sd'] for particle in overall_particles_per_iteration[iteration]])
            sd_iteration_sum = sum( [(particle['sd']/sd_iteration_max) for particle in overall_particles_per_iteration[iteration]])
            for idx, val in  enumerate(overall_particles_per_iteration[iteration]):
                if sd_iteration_sum >= 0:
                    coefficients[val['id']] += val['sd']/(sd_iteration_max * sd_iteration_sum* len(overall_particles_per_iteration))
        return coefficients

    def compute_diversity_coefficients_2(self, overall_particles_per_iteration):
        coefficients = defaultdict(int)
        for iteration in overall_particles_per_iteration:
            sd_iteration_sum = sum( [(particle['sd']) for particle in overall_particles_per_iteration[iteration]])
            for idx, val in  enumerate(overall_particles_per_iteration[iteration]):
                if sd_iteration_sum > 0:
                    coefficients[val['id']] += val['sd']/( sd_iteration_sum* len(overall_particles_per_iteration))
        return coefficients