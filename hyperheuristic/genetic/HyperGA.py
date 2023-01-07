import numpy
from pygad import GA, matplotlib


class HyperGA(GA):
    pass

    orchestrator = None
    best_actual_solutions = []

    def __init__(self,
                 num_generations,
                 num_parents_mating,
                 orchestrator,
                 initial_population=None,
                 sol_per_pop=None,
                 num_genes=None,
                 init_range_low=-4,
                 init_range_high=4,
                 gene_type=float,
                 parent_selection_type="sss",
                 keep_parents=-1,
                 K_tournament=3,
                 crossover_type="single_point",
                 crossover_probability=None,
                 mutation_type="random",
                 mutation_probability=None,
                 mutation_by_replacement=False,
                 mutation_percent_genes='default',
                 mutation_num_genes=None,
                 random_mutation_min_val=-1.0,
                 random_mutation_max_val=1.0,
                 gene_space=None,
                 allow_duplicate_genes=True,
                 on_start=None,
                 on_fitness=None,
                 on_parents=None,
                 on_crossover=None,
                 on_mutation=None,
                 callback_generation=None,
                 on_generation=None,
                 on_stop=None,
                 delay_after_gen=0.0,
                 save_best_solutions=False,
                 save_solutions=False,
                 suppress_warnings=False,
                 stop_criteria=None):
        GA.__init__(self,
                    num_generations,
                    num_parents_mating,
                    self.irrelevant_fitness_func,
                    initial_population,
                    sol_per_pop,
                    num_genes,
                    init_range_low,
                    init_range_high,
                    gene_type,
                    parent_selection_type,
                    keep_parents,
                    K_tournament,
                    crossover_type,
                    crossover_probability,
                    mutation_type,
                    mutation_probability,
                    mutation_by_replacement,
                    mutation_percent_genes,
                    mutation_num_genes,
                    random_mutation_min_val,
                    random_mutation_max_val,
                    gene_space,
                    allow_duplicate_genes,
                    on_start,
                    on_fitness,
                    on_parents,
                    on_crossover,
                    on_mutation,
                    callback_generation,
                    on_generation,
                    on_stop,
                    delay_after_gen,
                    save_best_solutions,
                    save_solutions,
                    suppress_warnings,
                    stop_criteria)
        self.orchestrator = orchestrator
        self.best_actual_solutions = []

    def irrelevant_fitness_func(solution, solution_idx):
        return 0

    def cal_pop_fitness(self):

        if self.valid_parameters == False:
            raise ValueError(
                "ERROR calling the cal_pop_fitness() method: \nPlease check the parameters passed while creating an instance of the GA class.\n")

        pop_fitness = []
        ensemble_best_particles, aptitude_coefficients, ensemble_global_particle = self.orchestrator.orchestrate(
            self.population)
        for sol_idx, sol in enumerate(self.population):
            pop_fitness.append(aptitude_coefficients[sol_idx])

        pop_fitness = numpy.array(pop_fitness)

        self.best_actual_solutions.append(ensemble_global_particle.value)

        return pop_fitness

    def plot_fitness(self,
                     title="HyperHeuristic - Generation vs. Fitness",
                     xlabel="Generation",
                     ylabel="Fitness",
                     linewidth=3,
                     font_size=14,
                     plot_type="plot",
                     color="#3870FF",
                     save_dir=None):
        fig = matplotlib.pyplot.figure()
        if plot_type == "plot":
            matplotlib.pyplot.plot(self.best_actual_solutions, linewidth=linewidth, color=color)

        matplotlib.pyplot.title(title, fontsize=font_size)
        matplotlib.pyplot.xlabel(xlabel, fontsize=font_size)
        matplotlib.pyplot.ylabel(ylabel, fontsize=font_size)

        if not save_dir is None:
            matplotlib.pyplot.savefig(fname=save_dir,
                                      bbox_inches='tight')
        matplotlib.pyplot.show()
