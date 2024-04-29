import numpy

from evaluation.Evaluator import Evaluator, np

from evaluation.functions.objective_functions import ackley
from hyperheuristic.DEHyperHeuristic import DEHyperHeuristic

# ESSA hyper-heuristic demonstrated on Ackley
from metaheuristic.triangle.TCOHeuristic import TCOHeuristic


problem = {'objective_func': ackley, 'dimensions': 10, 'bounds': (-32 * numpy.ones(10), 32 * numpy.ones(10)),
           'global_value': 0, 'name': "Ackley"}


# HyperDE Heuristic
DEHyperHeuristic().optimize(objective_func=problem['objective_func'],
                              dimensions=problem['dimensions'],
                             bounds=problem['bounds'])

# Centroid Flutter Search Heuristic
TCOHeuristic().optimize(objective_func=problem['objective_func'],
                              dimensions=problem['dimensions'],
                             bounds=problem['bounds'])

# Evaluate Multiple Methods
ranks = Evaluator().evaluate(from_file=False, nr_executions=60)
print(ranks)