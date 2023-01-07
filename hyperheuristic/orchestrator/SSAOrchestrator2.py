import concurrent.futures
from operator import attrgetter

import numpy as np

from hyperheuristic.agent.ssa.SSAAgent2 import SSAAgent2
from hyperheuristic.orchestrator.Orchestrator import Orchestrator
from hyperheuristic.orchestrator.metrics.AggregatedConsistencyMetric import AggregatedConsistencyMetric
from hyperheuristic.orchestrator.metrics.RelativeConvergenceMetric import RelativeConvergenceMetric


class SSAOrchestrator2(Orchestrator):
    convergence_metric = RelativeConvergenceMetric(set_divisor=3)
    consistency_metric = AggregatedConsistencyMetric(set_divisor=3)

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

            agent = SSAAgent2(id=id, problem=problem_dict1,
                              window_size=self.window_size, pop_size=self.n_quota_of_solutions, ST=options['ST'],
                              PD=options['PD'], SD=options['SD'])
            initial_solutions = self.tournament_selection(self.overall_solutions_state,
                                                          self.n_quota_of_solutions)
            agent.init_initial_positions(initial_solutions)
            agent_ensemble.append(agent)
        return agent_ensemble

    def orchestrate(self, population):
        ensemble_solutions_history = []
        ensemble_last_solutions = []
        agent_ensemble = self.compose(population)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(agent.solve): agent for agent in agent_ensemble}
            for fut in concurrent.futures.as_completed(futures):
                solutions_per_iteration = fut.result()
                ensemble_solutions_history.append(solutions_per_iteration)
                ensemble_last_solutions = np.append(ensemble_last_solutions,
                                                    solutions_per_iteration[self.window_size - 1])

        consistency_coefficient = self.consistency_metric.compute(ensemble_solutions_history, self.maximize)
        while consistency_coefficient < 0.40 and len(ensemble_solutions_history[0]) < 15:
            ensemble_solutions_history = []
            ensemble_last_solutions = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {executor.submit(agent.extend): agent for agent in agent_ensemble}
                for fut in concurrent.futures.as_completed(futures):
                    solutions_per_iteration = fut.result()
                    ensemble_solutions_history.append(solutions_per_iteration)
                    ensemble_last_solutions = np.append(ensemble_last_solutions,
                                                        solutions_per_iteration[self.window_size - 1])
            consistency_coefficient = self.consistency_metric.compute(ensemble_solutions_history, self.maximize)

        convergence_coefficients = self.convergence_metric.compute(ensemble_solutions_history, self.maximize)
        print(convergence_coefficients)
        print(consistency_coefficient)
        print(len(ensemble_solutions_history[0]))
        if self.maximize is True:
            ensemble_global_solution = max(ensemble_last_solutions, key=attrgetter('value'))
        else:
            ensemble_global_solution = min(ensemble_last_solutions, key=attrgetter('value'))

        print("Hypergeneration finished | best cost: {} best position: {}".format(ensemble_global_solution.value,
                                                                                  ensemble_global_solution.position))

        self.update_internal_state(ensemble_last_solutions, ensemble_global_solution)
        return ensemble_last_solutions, convergence_coefficients, ensemble_global_solution
