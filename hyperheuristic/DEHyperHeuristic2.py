from hyperheuristic.HyperHeuristic import HyperHeuristic
from hyperheuristic.genetic.HyperGA import HyperGA
from hyperheuristic.interceptor.Interceptor import inspect, plot_hyper_parameters
from hyperheuristic.orchestrator.DEOrchestrator import DEOrchestrator
from hyperheuristic.orchestrator.SSAOrchestrator import SSAOrchestrator


class DEHyperHeuristic2(HyperHeuristic):
    num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
    keep_parents = 5
    sol_per_pop = 10  # Number of solutions in the population.
    mutation_num_genes = 1

    def tournament_proportion_function(self, iteration):
        return  0.3 + iteration/39 * 0.3


    def optimize(self, objective_func, dimensions, bounds, nr_hypergenerations=40):
        gene_space = [{'low': 0.0, 'high': 1.0}, {'low': 0.0, 'high': 1.0}, range(0, 6)] #00 06 pare bun 1 rateu
        num_genes = 3
        window_size = 5
        orchestrator = DEOrchestrator(objective_func=objective_func, dimensions=dimensions, bounds=bounds,
                                       n_quota_of_particles=20, window_size=window_size, maximize=False)

        ga_instance = HyperGA(num_generations=nr_hypergenerations,
                              num_parents_mating=self.num_parents_mating,
                              keep_parents=self.keep_parents,
                              sol_per_pop=self.sol_per_pop,
                              num_genes=num_genes,
                              orchestrator=orchestrator,
                              on_generation=inspect,
                              gene_space=gene_space,
                              mutation_num_genes=self.mutation_num_genes,
                              # save_solutions=True
                              tournament_proportion_function=self.tournament_proportion_function
                              )
        ga_instance.run()
        # plot_hyper_parameters()
        # ga_instance.plot_genes(plot_type='scatter')
        return ga_instance.best_actual_solutions
