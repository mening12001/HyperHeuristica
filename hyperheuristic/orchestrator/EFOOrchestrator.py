import concurrent.futures
import copy
from operator import attrgetter

import numpy as np
import random

from hyperheuristic.agent.efo.EFOAgent import EFOAgent
from hyperheuristic.orchestrator.Orchestrator import Orchestrator
from hyperheuristic.orchestrator.metrics.RelativeDiversityMetric import RelativeDiversityMetric
from hyperheuristic.interceptor.Exchanger import Exchanger
from hyperheuristic.orchestrator.metrics.RelativeConvergenceMetric import RelativeConvergenceMetric


class EFOOrchestrator(Orchestrator):

    def compose(self, population, tournament_proportion=None):
        efo_ensemble = []
        for id, genome_agent in enumerate(population):
            options = {'r_rate': genome_agent[0], 'ps_rate': genome_agent[1],
                       'p_field': genome_agent[2], 'n_field': genome_agent[3]}

            lb, ub = self.bounds
            problem_dict1 = {
                "fit_func": self.objective_func,
                "lb": [lb[0], ] * self.dimensions,
                "ub": [ub[0], ] * self.dimensions,
                "minmax": "min",
                "log_to": None,  # 'console',
                "save_population": False,
            }

            efo_agent = EFOAgent(id=id, problem=problem_dict1, window_size=self.window_size,
                                 pop_size=self.n_quota_of_solutions, r_rate=options['r_rate'],
                                 ps_rate=options['ps_rate'], p_field=options['p_field'], n_field=options['n_field'])
            initial_solutions = self.tournament_selection(self.overall_solutions_state,
                                                          self.n_quota_of_solutions)
            efo_agent.init_initial_positions(initial_solutions)
            efo_ensemble.append(efo_agent)
        return efo_ensemble
