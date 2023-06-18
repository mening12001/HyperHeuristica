# !/usr/bin/env python
# Created by "Thieu" at 17:22, 29/05/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
from copy import deepcopy

from hyperheuristic.agent.optimizer import Optimizer


class DEAgent(Optimizer):
    """
    The original version of: Differential Evolution (DE)

    Links:
        1. https://doi.org/10.1016/j.swevo.2018.10.006

    Hyper-parameters should fine tuned in approximate range to get faster convergen toward the global optimum:
        + wf (float): [0.5, 0.95], weighting factor, default = 0.8
        + cr (float): [0.5, 0.95], crossover rate, default = 0.9
        + strategy (int): [0, 5], there are lots of variant version of DE algorithm,
            + 0: DE/rand/1/bin
            + 1: DE/best/1/bin
            + 2: DE/best/2/bin
            + 3: DE/rand/2/bin
            + 4: DE/current-to-best/1/bin
            + 5: DE/current-to-rand/1/bin

    Examples
    ~~~~~~~~
    >>> import numpy as np
    >>> from mealpy.evolutionary_based.DE import BaseDE
    >>>
    >>> def fitness_function(solution):
    >>>     return np.sum(solution**2)
    >>>
    >>> problem_dict1 = {
    >>>     "fit_func": fitness_function,
    >>>     "lb": [-10, -15, -4, -2, -8],
    >>>     "ub": [10, 15, 12, 8, 20],
    >>>     "minmax": "min",
    >>> }
    >>>
    >>> epoch = 1000
    >>> pop_size = 50
    >>> wf = 0.7
    >>> cr = 0.9
    >>> strategy = 0
    >>> model = BaseDE(problem_dict1, epoch, pop_size, wf, cr, strategy)
    >>> best_position, best_fitness = model.solve()
    >>> print(f"Solution: {best_position}, Fitness: {best_fitness}")

    References
    ~~~~~~~~~~
    [1] Mohamed, A.W., Hadi, A.A. and Jambi, K.M., 2019. Novel mutation strategy for enhancing SHADE and
    LSHADE algorithms for global numerical optimization. Swarm and Evolutionary Computation, 50, p.100455.
    """

    def __init__(self, id, problem, epoch=10000, pop_size=100, wf=0.8, cr=0.9, strategy=0, **kwargs):
        """
        Args:
            problem (dict): The problem dictionary
            epoch (int): maximum number of iterations, default = 10000
            pop_size (int): number of population size, default = 100
            wf (float): weighting factor, default = 0.8
            cr (float): crossover rate, default = 0.9
            strategy (int): Different variants of DE, default = 0
        """
        super().__init__(id, problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.wf = self.validator.check_float("wf", float(wf), (0, 2.0))
        self.cr = self.validator.check_float("cr", float(cr), (0, 1.0))
        self.strategy = self.validator.check_int("strategy", int(strategy), [0, 6])

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False


    def _mutation__(self, current_pos, new_pos):
        pos_new = np.where(np.random.uniform(0, 1, self.problem.n_dims) < self.cr, current_pos, new_pos)
        return self.amend_position(pos_new, self.problem.lb, self.problem.ub)

    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop = []
        if self.strategy == 0:
            # Choose 3 random element and different to i
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx_list[0]][self.ID_POS] + self.wf * \
                          (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 1:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.g_best[self.ID_POS] + self.wf * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 2:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 4, replace=False)
                pos_new = self.g_best[self.ID_POS] + self.wf * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS]) + \
                          self.wf * (self.pop[idx_list[2]][self.ID_POS] - self.pop[idx_list[3]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 3:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 5, replace=False)
                pos_new = self.pop[idx_list[0]][self.ID_POS] + self.wf * \
                          (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS]) + \
                          self.wf * (self.pop[idx_list[3]][self.ID_POS] - self.pop[idx_list[4]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 4:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + self.wf * (self.g_best[self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                          self.wf * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[1]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 5:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                pos_new = self.pop[idx][self.ID_POS] + self.wf * (self.pop[idx_list[0]][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                          self.wf * (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        elif self.strategy == 6:
            for idx in range(0, self.pop_size):
                idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                temp_pop = [self.pop[idx_list[0]], self.pop[idx_list[1]], self.pop[idx_list[2]]]
                if self.problem.minmax == "min":
                    sorted_pop = sorted(temp_pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
                else:
                    sorted_pop = sorted(temp_pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
                pos_new = self.pop[idx][self.ID_POS] + self.wf * (
                            sorted_pop[0][self.ID_POS] - self.pop[idx][self.ID_POS]) + \
                          self.wf * (sorted_pop[1][self.ID_POS] - sorted_pop[2][self.ID_POS])
                pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                pop.append([pos_new, None])
        else:
                    for idx in range(0, self.pop_size):
                        idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)
                        if self.pop[idx_list[0]][self.ID_TAR][self.ID_FIT] < self.pop[idx_list[1]][self.ID_TAR][self.ID_FIT]:
                            pos_new = self.pop[idx_list[0]][self.ID_POS] + self.wf * \
                                      (self.pop[idx_list[1]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS])
                        else:
                            pos_new = self.pop[idx_list[1]][self.ID_POS] + self.wf * (
                                    self.pop[idx_list[0]][self.ID_POS] - self.pop[idx_list[2]][self.ID_POS])
                        pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
                        pop.append([pos_new, None])

        pop = self.update_target_wrapper_population(pop)

        # create new pop by comparing fitness of corresponding each member in pop and children
        self.pop = self.greedy_selection_population(self.pop, pop)


