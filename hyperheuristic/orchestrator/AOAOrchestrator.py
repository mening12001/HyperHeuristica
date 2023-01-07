from hyperheuristic.agent.aoa.AOAAgent import AOAAgent
from hyperheuristic.orchestrator.Orchestrator import Orchestrator


class AOAOrchestrator(Orchestrator):

    def __init__(self, objective_func, dimensions, bounds, n_quota_of_particles,
                 window_size, overall_nr_iterations,
                 maximize=True):
        super().__init__(objective_func, dimensions, bounds, n_quota_of_particles, window_size, maximize)
        self.overall_nr_iterations = overall_nr_iterations

    def compose(self, population):
        aoa_ensemble = []
        for id, genome_agent in enumerate(population):
            options = {'alpha': genome_agent[0], 'miu': genome_agent[1]
                , 'moa_min': genome_agent[2], 'moa_max': genome_agent[3]}

            lb, ub = self.bounds
            problem_dict1 = {
                "fit_func": self.objective_func,
                "lb": [lb[0], ] * self.dimensions,
                "ub": [ub[0], ] * self.dimensions,
                "minmax": "min",
                "log_to": None,  # 'console',
                "save_population": False,
            }
            aoa_agent = AOAAgent(id=id, overall_nr_iterations=self.overall_nr_iterations, problem=problem_dict1, window_size=self.window_size,
                                 pop_size=self.n_quota_of_solutions, alpha=options['alpha'], miu=options['miu'],
                                 moa_min=options['moa_min'], moa_max=options['moa_max'])
            initial_solutions = self.tournament_selection(self.overall_solutions_state,
                                                          self.n_quota_of_solutions)
            aoa_agent.init_initial_positions(initial_solutions)
            aoa_ensemble.append(aoa_agent)
        return aoa_ensemble
