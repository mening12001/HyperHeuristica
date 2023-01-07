from hyperheuristic.agent.bes.BESAgent import BESAgent
from hyperheuristic.orchestrator.Orchestrator import Orchestrator


class BESOrchestrator(Orchestrator):

    def compose(self, population):
        bes_ensemble = []
        for id, genome_agent in enumerate(population):
            options = {'a_factor': genome_agent[0], 'R_factor': genome_agent[1],
                       'alpha': genome_agent[2], 'c1': genome_agent[3], 'c2': genome_agent[4]}

            lb, ub = self.bounds
            problem_dict1 = {
                "fit_func": self.objective_func,
                "lb": [lb[0], ] * self.dimensions,
                "ub": [ub[0], ] * self.dimensions,
                "minmax": "min",
                "log_to": None,  # 'console',
                "save_population": False,
            }

            bes_agent = BESAgent(id=id, problem=problem_dict1,
                                 window_size=self.window_size, pop_size=self.n_quota_of_solutions,
                                 a_factor=options['a_factor'], R_factor=options['R_factor'], alpha=options['alpha'],
                                 c1=options['c1'], c2=options['c2'])
            initial_solutions = self.tournament_selection(self.overall_solutions_state, self.n_quota_of_solutions)
            bes_agent.init_initial_positions(initial_solutions)
            bes_ensemble.append(bes_agent)
        return bes_ensemble


