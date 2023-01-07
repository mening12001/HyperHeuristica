import numpy

from evaluation.Evaluator import Evaluator
from evaluation.functions.objective_functions import ackley
from hyperheuristic.ESSAHyperHeuristic import ESSAHyperHeuristic

# ESSA hyper-heuristic demonstrated on Ackley
problem = {'objective_func': ackley, 'dimensions': 10, 'bounds': (-32 * numpy.ones(10), 32 * numpy.ones(10)),
           'global_value': 0, 'name': "Ackley"}

ESSAHyperHeuristic().optimize(objective_func=problem['objective_func'],
                              dimensions=problem['dimensions'],
                              bounds=problem['bounds'])

# Evaluation on CEC problems of the methods considered
# Evaluator().evaluate()
