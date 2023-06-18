import concurrent.futures
import copy
from operator import attrgetter

import numpy as np
import random

from hyperheuristic.orchestrator.metrics.RelativeDiversityMetric import RelativeDiversityMetric
from hyperheuristic.interceptor.Exchanger import Exchanger
from hyperheuristic.orchestrator.metrics.RelativeConvergenceMetric import RelativeConvergenceMetric
from hyperheuristic.agent.pso.PSOAgent import PSOAgent

class PSOOrchestrator:

    n_quota_of_particles = 0
    dimensions = 0
    objective_func = 0
    window_size = 0
    covergence_metric = RelativeConvergenceMetric(set_divisor=3)
    diversity_metric = RelativeDiversityMetric()
    maximize = True
    overall_particles_state = None
    overall_global_particle_state = None
    bounds = None
    iteration = 0

    def __init__(self, objective_func, dimensions, bounds, n_quota_of_particles, window_size, maximize=True):
        self.dimensions = dimensions
        self.objective_func = objective_func
        self.n_quota_of_particles = n_quota_of_particles
        self.window_size = window_size
        self.maximize = maximize
        self.bounds = bounds


    def compose(self, population, tournament_size=None):
        pso_ensemble = []
        for id, genome_agent in enumerate(population):
            options = {'c1': genome_agent[0], 'c2': genome_agent[1]#, 'c3': genome_agent[2]
                , 'w': genome_agent[2]}
            pso_agent = PSOAgent(id, n_particles=self.n_quota_of_particles, dimensions=self.dimensions, bounds=self.bounds, options=options)
            initial_particles = self.tournament_selection_2( self.overall_particles_state, self.n_quota_of_particles)#, len(population))
            global_particle = copy.deepcopy(self.overall_global_particle_state)
            pso_agent.set_initial_particles_state(initial_particles)
            #pso_agent.set_global_particle_state(global_particle)
            pso_ensemble.append(pso_agent)
        return pso_ensemble

    def orchestrate(self, population):
        ensemble_particles_history = []
        ensemble_last_particles = []
        ensemble_global_particles = []
        pso_ensemble = self.compose(population)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(pso.optimize, self.objective_func, self.window_size): pso for pso in pso_ensemble}
            for fut in concurrent.futures.as_completed(futures):
                particles_per_iteration, best_values_per_iteration, best_particle = fut.result()
                ensemble_particles_history.append(particles_per_iteration)
                ensemble_last_particles = np.append(ensemble_last_particles, particles_per_iteration[self.window_size - 1])
                ensemble_global_particles = np.append(ensemble_global_particles, best_particle)

        convergence_coefficients = self.covergence_metric.compute(ensemble_particles_history, self.maximize)

        if self.maximize is True:
            ensemble_global_particle = max(ensemble_global_particles, key=attrgetter('value'))
        else:
            ensemble_global_particle = min(ensemble_global_particles, key=attrgetter('value'))

        print("Hypergeneration finished | best cost: {} ssa position: {}".format(ensemble_global_particle.value, ensemble_global_particle.position))

        self.update_internal_state(ensemble_last_particles, ensemble_global_particle)
        return  (ensemble_last_particles,  convergence_coefficients, ensemble_global_particle)

    def update_internal_state(self, ensemble_last_particles, ensemble_global_particle):
        self.overall_particles_state = ensemble_last_particles
        self.overall_global_particle_state = ensemble_global_particle

    def tournament_selection(self, ensemble_particles, n_quota_of_particles):
        if ensemble_particles is None:
            return None
        temp_ensemble_particles = copy.deepcopy(ensemble_particles)
        best_ensemble_particles = sorted(temp_ensemble_particles, key=lambda p: p.value, reverse=self.maximize)[:len(ensemble_particles)//2]
        selected_particles = []
        for i in range(n_quota_of_particles):
            selected = random.choices(best_ensemble_particles, k=5)
            selected = sorted(selected, key=lambda p: p.value, reverse=self.maximize)
            selected_particles.append(selected[0])
        return selected_particles

    def tournament_selection_2(self, ensemble_particles, n_quota_of_particles):
        if ensemble_particles is None:
            return None
        temp_ensemble_particles = copy.deepcopy(ensemble_particles)
        selected = random.choices(temp_ensemble_particles, k=len( ensemble_particles)//2)
        selected = sorted(selected, key=lambda p: p.value, reverse=self.maximize)[:n_quota_of_particles]
        return selected

    def tournament_selection_3(self, id, ensemble_particles, n_quota_of_particles, pop_size):
        if ensemble_particles is None:
            return None
        temp_ensemble_particles = copy.deepcopy(ensemble_particles)
        idx = []
        if id - 1 < 0:
            idx.append(pop_size-1 - (id ))
        else:
            idx.append(id-1)
        idx.append(id)

        if id + 1 > pop_size - 1:
            idx.append(id - pop_size +1)
        else:
            idx.append(id + 1)

        if id + 2 > pop_size - 1:
            idx.append(id - pop_size +2)
        else:
            idx.append(id + 2)

        if id - 2 < 0:
            idx.append(pop_size - 2 - (id))
        else:
            idx.append(id - 2)


        #idx = random.sample(range(0, pop_size), 2)

        selected = list(filter(lambda p: idx.__contains__(p.id), temp_ensemble_particles))


        #selected = random.choices(temp_ensemble_particles, k=len( ensemble_particles)//2)
        selected = sorted(selected, key=lambda p: p.value, reverse=self.maximize)[:n_quota_of_particles]
        return selected

    def makeWheel(self, population):

        temp_population = copy.deepcopy(population)
        min_value = min(p.value for p in temp_population)
        if min_value < 0:
            for p in temp_population:
                p.value = p.value + abs(min_value)

        if self.maximize is False:
            for p in temp_population:
                p.value = 100/(p.value + 1)

        wheel = []
        total = sum(p.value for p in temp_population)
        top = 0
        for i, p in enumerate(temp_population):
            f = p.value / total
            wheel.append((top, top + f, population[i]))
            top += f
        return wheel

    def binSearch(self, wheel, num):
        mid = len(wheel) // 2
        low, high, answer = wheel[mid]
        if low <= num <= high:
            return answer
        elif high < num:
            return self.binSearch(wheel[mid + 1:], num)
        else:
            return self.binSearch(wheel[:mid], num)

    def select(self, population, N):
        if population is None:
            return None

        population = sorted(population, key=lambda p: p.value, reverse=self.maximize)[:len(population) // 3]
        random.shuffle(population)
        wheel = self.makeWheel(population)
        stepSize = 1.0 / N
        answer = []
        r = random.random()
        answer.append(self.binSearch(wheel, r))
        while len(answer) < N:
            r += stepSize
            if r > 1:
                r %= 1
            answer.append(self.binSearch(wheel, r))
        return answer

