

from hyperheuristic.HyperHeuristic import HyperHeuristic
from hyperheuristic.genetic.HyperGA import HyperGA
from hyperheuristic.orchestrator.AOAOrchestrator import AOAOrchestrator
from hyperheuristic.orchestrator.BESOrchestrator import BESOrchestrator
from hyperheuristic.orchestrator.PSOOrchestrator import PSOOrchestrator
from hyperheuristic.orchestrator.SSAOrchestrator import SSAOrchestrator


class GenericHyperHeuristic(HyperHeuristic):
    num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
    keep_parents = 5
    sol_per_pop = 10  # Number of solutions in the population.
    mutation_num_genes = 1

    def __init__(self, orchestrator='PSO'):
        self.orchestrator = orchestrator

    def optimize(self, objective_func, dimensions, bounds, nr_hypergenerations=40):
        orchestrator = None
        if self.orchestrator == 'AOA':
            gene_space = [range(2, 10), {'low': 0.1, 'high': 2.0}
                , {'low': 0, 'high': 0.41}, {'low': 0.41, 'high': 1.0}]
            num_genes = 4
            window_size = 5
            orchestrator = AOAOrchestrator(objective_func=objective_func, dimensions=dimensions, bounds=bounds,
                                           n_quota_of_particles=20, window_size=window_size,
                                           overall_nr_iterations=window_size * nr_hypergenerations, maximize=False)
        elif self.orchestrator == 'SSA':
            gene_space = [{'low': 0.5, 'high': 1.0}, {'low': 0.0, 'high': 1.0}
                , {'low': 0.0, 'high': 1.0}]
            num_genes = 3
            window_size = 5
            orchestrator = SSAOrchestrator(objective_func=objective_func, dimensions=dimensions, bounds=bounds,
                                           n_quota_of_particles=20, window_size=window_size,
                                           overall_nr_iterations=window_size * nr_hypergenerations, maximize=False)
        elif self.orchestrator == 'BES':
            gene_space = [range(2, 20), {'low': 0.1, 'high': 3.0}
                , {'low': 0.5, 'high': 3.0}, {'low': 0.0, 'high': 4.0}, {'low': 0.0, 'high': 4.0}]
            num_genes = 5
            orchestrator = BESOrchestrator(objective_func=objective_func, dimensions=dimensions, bounds=bounds,
                                           n_quota_of_particles=20, window_size=5, maximize=False)
        elif self.orchestrator == 'PSO':
            gene_space = [{'low': 0.0, 'high': 4.0}, {'low': 0.0, 'high': 4.0}
                , {'low': 0.0, 'high': 1.0}]
            num_genes = 3
            orchestrator = PSOOrchestrator(objective_func=objective_func, dimensions=dimensions, bounds=bounds,
                                           n_quota_of_particles=20, window_size=5, maximize=False)

        ga_instance = HyperGA(num_generations=nr_hypergenerations,
                              num_parents_mating=self.num_parents_mating,
                              keep_parents=self.keep_parents,
                              sol_per_pop=self.sol_per_pop,
                              num_genes=num_genes,
                              orchestrator=orchestrator,
                              # on_generation=on_generation,
                              gene_space=gene_space,
                              mutation_num_genes=self.mutation_num_genes)
        ga_instance.run()
        return ga_instance.best_actual_solutions
