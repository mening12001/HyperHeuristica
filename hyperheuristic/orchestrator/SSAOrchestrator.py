from hyperheuristic.agent.ssa.SSAAgent import SSAAgent
from hyperheuristic.orchestrator.Orchestrator import Orchestrator


class SSAOrchestrator(Orchestrator):

    def __init__(self, objective_func, dimensions, bounds, n_quota_of_particles,
                 window_size, maximize=True):
        super().__init__(objective_func, dimensions, bounds, n_quota_of_particles, window_size, maximize)

    def compose(self, population):
        agent_ensemble = []
        for id, genome_agent in enumerate(population):
            options = {'ST': genome_agent[0], 'PD': genome_agent[1]
                , 'SD': genome_agent[2]}

            lb, ub = self.bounds
            problem_dict1 = {
                "fit_func": self.objective_func,
                "lb": [lb[0], ] * self.dimensions,
                "ub": [ub[0], ] * self.dimensions,
                "minmax": "min",
                "log_to": None,  # 'console',
                "save_population": False,
            }

            agent = SSAAgent(id=id, problem=problem_dict1,
                             window_size=self.window_size, pop_size=self.n_quota_of_solutions, ST=options['ST'],
                             PD=options['PD'], SD=options['SD'])
            initial_solutions = self.tournament_selection(self.overall_solutions_state,
                                                          self.n_quota_of_solutions)
            agent.init_initial_positions(initial_solutions)
            agent_ensemble.append(agent)
        return agent_ensemble
