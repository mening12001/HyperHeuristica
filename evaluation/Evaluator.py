import json

import numpy
from opfunu.cec_based import F42022

from evaluation.Trial import Trial
from evaluation.functions.cec_objective_functions import F102022, F102022_10, F112022_10, F12022_10, F72021_10, \
    F82022_10, F12020_5, F42020_10, F72022_10, F92022_10, F22022_10, F32022_10, F42022_10, F52022_10, F62022_10, \
    F122022_10
from hyperheuristic.DEHyperHeuristic import DEHyperHeuristic
from hyperheuristic.ESSAHyperHeuristic import ESSAHyperHeuristic
from hyperheuristic.EBESHyperHeuristic import EBESHyperHeuristic

from evaluation.functions.objective_functions import *
from metaheuristic.DE.DEHeuristic import DEHeuristic
from metaheuristic.DE.LSHADEHeuristic import LSHADEHeuristic
from metaheuristic.DE.SHADEHeuristic import SHADEHeuristic
from metaheuristic.abc.ABCHeuristic import ABCHeuristic
from metaheuristic.acor.ACORHeuristic import ACORHeuristic
from metaheuristic.ao.AOHeuristic import AOHeuristic
from metaheuristic.bes.BESHeuristic import BESHeuristic
from metaheuristic.cgo.CGOHeuristic import CGOHeuristic
from metaheuristic.cro.CROHeuristic import CROHeuristic
from metaheuristic.fa.FAHeuristic import FAHeuristic
from metaheuristic.ffa.FFAHeuristic import FFAHeuristic
from metaheuristic.fpa.FPAHeuristic import FPAHeuristic
from metaheuristic.ga.GAHeuristic import GAHeuristic
from metaheuristic.hgs.HGSHeuristic import HGSHeuristic
from metaheuristic.mfo.MFOHeuristic import MFOHeuristic
from metaheuristic.pso.PSOHeuristic import PSOHeuristic
from metaheuristic.sca.SCAHeuristic import SCAHeuristic
from metaheuristic.sma.SMAHeuristic import SMAHeuristic
from metaheuristic.aoa.AOAHeuristic import AOAHeuristic
from metaheuristic.hho.HHOHeuristic import HHOHeuristic
from metaheuristic.ssa.SSAHeuristic import SSAHeuristic
from metaheuristic.woa.WOAHeuristic import WOAHeuristic


class Evaluator:
    problems = [
        {'objective_func': ackley, 'dimensions': 10, 'bounds': (-32 * numpy.ones(10), 32 * numpy.ones(10)),
         'global_value': 0, 'name': "Ackley"},
        {'objective_func': beale, 'dimensions': 2, 'bounds': (-4.5 * numpy.ones(2), 4.5 * numpy.ones(2)),
         'global_value': 0, 'name': "Beale"},
        {'objective_func': booth, 'dimensions': 2, 'bounds': (-10 * numpy.ones(2), 10 * numpy.ones(2)),
         'global_value': 0, 'name': "Booth"},
        #    {'objective_func': fx.bukin6, 'dimensions': 2, 'bounds': (-10 * numpy.ones(2), 10 * numpy.ones(2))},
        {'objective_func': crossintray, 'dimensions': 2, 'bounds': (-10 * numpy.ones(2), 10 * numpy.ones(2)),
         'global_value': -2.06261, 'name': "Crossintray"},
        {'objective_func': easom, 'dimensions': 2, 'bounds': (-100 * numpy.ones(2), 100 * numpy.ones(2)),
         'global_value': -1, 'name': "Easom"},
        {'objective_func': eggholder, 'dimensions': 2, 'bounds': (-512 * numpy.ones(2), 512 * numpy.ones(2)),
         'global_value': -959.6407, 'name': "Eggholder"},
        {'objective_func': goldstein, 'dimensions': 2, 'bounds': (-2 * numpy.ones(2), 2 * numpy.ones(2)),
         'global_value': 3, 'name': "Goldstein"},
        {'objective_func': himmelblau, 'dimensions': 2, 'bounds': (-5 * numpy.ones(2), 5 * numpy.ones(2)),
         'global_value': 0, 'name': "Himmelblau"},
        {'objective_func': holdertable, 'dimensions': 2, 'bounds': (-10 * numpy.ones(2), 10 * numpy.ones(2)),
         'global_value': -19.2085, 'name': "Holdertable"},
        {'objective_func': levi, 'dimensions': 2, 'bounds': (-10 * numpy.ones(2), 10 * numpy.ones(2)),
         'global_value': 0, 'name': "Levi"},
        {'objective_func': matyas, 'dimensions': 2, 'bounds': (-10 * numpy.ones(2), 10 * numpy.ones(2)),
         'global_value': 0, 'name': "Matyas"},
        {'objective_func': rastrigin, 'dimensions': 10, 'bounds': (-5.12 * numpy.ones(10), 5.12 * numpy.ones(10)),
         'global_value': 0, 'name': "Rastrigin"},
        {'objective_func': rosenbrock, 'dimensions': 10, 'bounds': (-2.048 * numpy.ones(10), 2.048 * numpy.ones(10)),
         'global_value': 0, 'name': "Rosenbrock"},
        {'objective_func': schaffer2, 'dimensions': 2, 'bounds': (-100 * numpy.ones(2), 100 * numpy.ones(2)),
         'global_value': 0, 'name': "Schaffer 2"},
        {'objective_func': threehump, 'dimensions': 2, 'bounds': (-5 * numpy.ones(2), 5 * numpy.ones(2)),
         'global_value': 0, 'name': "Threehump"},
        {'objective_func': griewank, 'dimensions': 10, 'bounds': (-600 * numpy.ones(10), 600 * numpy.ones(10)),
         'global_value': 0, 'name': "Griewank"},
        {'objective_func': dropwave, 'dimensions': 2, 'bounds': (-5.12 * numpy.ones(2), 5.12 * numpy.ones(2)),
         'global_value': -1, 'name': "Dropwave"},
        {'objective_func': shubert, 'dimensions': 2, 'bounds': (-5.12 * numpy.ones(2), 5.12 * numpy.ones(2)),
         'global_value': -186.7309, 'name': "Shubert"},
        {'objective_func': bohachevsky1, 'dimensions': 2, 'bounds': (-100 * numpy.ones(2), 100 * numpy.ones(2)),
         'global_value': 0, 'name': "Bohachevsky 1"},
        {'objective_func': styblinskitang, 'dimensions': 10, 'bounds': (-5 * numpy.ones(10), 5 * numpy.ones(10)),
         'global_value': -39.16599 * 10, 'name': "Styblinskitang"}
    ]

    problems = [
        {'objective_func': F102022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 2400, 'name': "F10: Composition Function 2"},
        {'objective_func': F112022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 2600, 'name': "F11: Composition Function 3"},
        {'objective_func': F12022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 300, 'name': "F1: Shifted and full Rotated Zakharov Function"},
        {'objective_func': F72021_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 2100, 'name': "F7: Hybrid Function 3"},
        {'objective_func': F82022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 2200, 'name': "F8: Hybrid Function 3"},
        {'objective_func': F12020_5, 'dimensions': 5, 'bounds': (-100 * numpy.ones(5), 100 * numpy.ones(5)),
         'global_value': 100, 'name': "F1: Shifted and Rotated Bent Cigar Function"},
        {'objective_func': F42020_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 1900, 'name': "F4: Expanded Rosenbrock’s plus Griewank’s Function"},
        {'objective_func': F72022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 2000, 'name': "F7: Hybrid Function 2"},
        {'objective_func': F92022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 2300, 'name': "F9: Composition Function 1"},
        {'objective_func': F22022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
         'global_value': 400, 'name': "F2: Shifted and Rotated Rosenbrock’s Function"}
    ]
    problems = [{'objective_func': F32022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 600, 'name': "F3: Shifted and full Rotated Expanded Schaffer’s F7"},
                {'objective_func': F42022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 800, 'name': "F4: Shifted and Rotated Non-Continuous Rastrigin’s Function"},
                {'objective_func': F52022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 900, 'name': "F5: Shifted and Rotated Levy Function"},
                {'objective_func': F62022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 1800, 'name': "F6: Hybrid Function 1"},
                {'objective_func': F122022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 2700, 'name': "F12: Composition Function 4"},
                {'objective_func': F102022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 2400, 'name': "F10: Composition Function 2"},
                {'objective_func': F112022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 2600, 'name': "F11: Composition Function 3"},
                {'objective_func': F12022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 300, 'name': "F1: Shifted and full Rotated Zakharov Function"},
                {'objective_func': F82022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 2200, 'name': "F8: Hybrid Function 3"},
                {'objective_func': F72022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 2000, 'name': "F7: Hybrid Function 2"},
                {'objective_func': F92022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 2300, 'name': "F9: Composition Function 1"},
                {'objective_func': F22022_10, 'dimensions': 10, 'bounds': (-100 * numpy.ones(10), 100 * numpy.ones(10)),
                 'global_value': 400, 'name': "F2: Shifted and Rotated Rosenbrock’s Function"}
                ]

    heuristics = [ESSAHyperHeuristic(), HHOHeuristic(), HGSHeuristic(), SSAHeuristic(), BESHeuristic(), SMAHeuristic(),
                  CGOHeuristic(),
                  WOAHeuristic(), SCAHeuristic(), AOHeuristic(), FFAHeuristic(), ACORHeuristic(), PSOHeuristic(),
                  CROHeuristic(), AOAHeuristic(),
                  DEHeuristic(), ABCHeuristic(), FPAHeuristic(), GAHeuristic(), MFOHeuristic(), FAHeuristic()]

    heuristics = [ESSAHyperHeuristic(), SSAHeuristic(), WOAHeuristic(), CGOHeuristic(), BESHeuristic(), HHOHeuristic(),
                  HGSHeuristic(),
                  SMAHeuristic(), DEHeuristic(), GAHeuristic(), SCAHeuristic(), AOHeuristic(), ACORHeuristic(),
                  ABCHeuristic(), FPAHeuristic(), PSOHeuristic()]

    heuristics = [DEHyperHeuristic(), ESSAHyperHeuristic(), SSAHeuristic(), SHADEHeuristic(), LSHADEHeuristic(), DEHeuristic(),
                  EBESHyperHeuristic(), BESHeuristic(), WOAHeuristic(), CGOHeuristic(), HHOHeuristic(),
                  HGSHeuristic(), SMAHeuristic()]



    def evaluate(self, from_file=False, verbose=True, nr_executions=10):
        trials = []
        results = {heuristic.__class__.__name__: 0 for heuristic in self.heuristics}
        for problem in self.problems:
            trials.append(Trial(problem=problem, heuristics=self.heuristics, nr_executions=nr_executions))
        for trial in trials:
            if verbose is True:
                rank_results = trial.obtain_rank_verbose_from_file() if from_file is True else trial.obtain_rank_verbose()
            else:
                rank_results = trial.obtain_rank_from_file() if from_file is True else trial.obtain_rank()
            print(trial.problem['objective_func'].__name__,
                  dict(sorted(rank_results.items(), key=lambda item: item[1])))
            for key in rank_results:
                results[key] += rank_results[key] / len(trials)
        ranking = dict(sorted(results.items(), key=lambda item: item[1]))
        with open('results.json', 'w') as outfile:
            json.dump(ranking, outfile)
        return ranking

    def friedman_ranking_from_file(self, significance=0.0000001,  nr_executions=10):
        trials = []
        results = {heuristic.__class__.__name__: 0 for heuristic in self.heuristics}
        for problem in self.problems:
            trials.append(Trial(problem=problem, heuristics=self.heuristics, nr_executions=nr_executions))
        for trial in trials:
            rank_results = trial.obtain_median_rank_verbose_from_file()
            values = sorted(rank_results.values())
            sorted_methods = sorted(rank_results.items(), key=lambda x: x[1])
            start_index = 0
            index_sum = 1
            for index in range(1, len(values)):
                if values[index] - values[index - 1] <= significance:
                    index_sum += (index + 1)
                else:
                    rank = index_sum/(index - start_index)
                    index_sum = index
                    for index2 in range(start_index, index):
                        results[sorted_methods[index2][0]] += rank/len(trials)
                    start_index = index
        ranking = dict(sorted(results.items(), key=lambda item: item[1]))
        with open('friedman_results.json', 'w') as outfile:
            json.dump(ranking, outfile)
        return ranking

    def plot_convergence_from_file(self, nr_executions=10):
        for problem in self.problems:
            trial = Trial(problem=problem, heuristics=self.heuristics, nr_executions=nr_executions)
            trial.plot_convergence()

    def plot_box_from_file(self, verbose=True, nr_executions=10):
        for problem in self.problems:
            trial = Trial(problem=problem, heuristics=self.heuristics, nr_executions=nr_executions)
            trial.plot_box(verbose=verbose)
