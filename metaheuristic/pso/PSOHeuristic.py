from mealpy.math_based import AOA

from metaheuristic.hho import HHO
from metaheuristic.mfo import MFO
from metaheuristic.pso import PSO

class PSOHeuristic:

    def optimize(self, objective_func, dimensions, bounds, nr_iterations=200):
        lb, ub = bounds
        problem_dict1 = {
            "fit_func": objective_func,
            "lb": [lb[0], ] * dimensions,
            "ub": [ub[0], ] * dimensions,
            "minmax": "min",
            "log_to": 'console',
            "save_population": False,
        }
        model = PSO.BasePSO(problem_dict1, epoch=nr_iterations, pop_size=200)
        return model.solve()