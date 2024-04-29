# !/usr/bin/env python
# Created by "Thieu" at 09:48, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%
import math

from matplotlib import pyplot
from noise import pnoise3, pnoise1

import numpy as np
from scipy.stats import cauchy
from copy import deepcopy
from opfunu.cec_based import F102022, F112022, F12022, F72021, F82022, F12020, F42020, F72022, F92022, F22022, F32022, \
    F42022, F52022, F62022, F122022
from metaheuristic.optimizer.optimizer import Optimizer

class TCO(Optimizer):


    def __init__(self, problem, epoch=10000, pop_size=100, sp=0.8, alpha=1.5, mp=0.5, **kwargs):

        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.sp = self.validator.check_float("sp", sp, (0, 1))
        self.alpha = self.validator.check_float("alpha", alpha, (1, 2))
        self.mp = self.validator.check_float("mp", mp, (0, 1))

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def perlin(self, current_position):
        perlin_values = []
        for i in range(self.problem.n_dims):
            perlin_values.append(
                pnoise1(current_position[i]))  # Use pnoise2() or pnoise3() for 2D or 3D noise respectively
        return perlin_values

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
        pop_sorted = self.get_sorted_strim_population(self.pop)

        for idx in range(0, self.pop_size):
            idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)


            centroid = (pop_sorted[0][self.ID_POS] + pop_sorted[1][self.ID_POS] +pop_sorted[2][self.ID_POS]) / 3
            centroid2 = (self.pop[idx_list[2]][self.ID_POS]  + self.pop[idx_list[0]][self.ID_POS] + self.pop[idx_list[1]][self.ID_POS] ) / 3

            if np.random.uniform(0, 1) < self.sp:
                pos_new = self.pop[idx][self.ID_POS] + np.random.binomial(n=1, p=self.mp, size=[self.problem.n_dims]) *  self.levy_step() * (centroid  - self.pop[idx][self.ID_POS])
            else:
                pos_new = self.pop[idx][self.ID_POS] +  self.perlin(self.pop[idx][self.ID_POS]) * (centroid2 - self.pop[idx][self.ID_POS])

            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])

        pop = self.update_target_wrapper_population(pop)
        self.pop = self.greedy_selection_population(self.pop, pop)

    def levy_step(self):
        alpha = self.alpha#1.5  # Parameter controlling the tail heaviness (typically > 1)
        sigma = (math.gamma(1 + alpha) * np.sin(math.pi * alpha / 2) / (
                    math.gamma((1 + alpha) / 2) * alpha * (2 ** ((alpha - 1) / 2)))) ** (1 / alpha)

        # Generate a step length
        u = np.random.uniform(0,1)  # Random number from a uniform distribution [0, 1]
        v = np.random.uniform(0, 1)  # Random number from a uniform distribution [0, 1]

        return (u / abs(v) ** (1 / alpha)) * sigma











class TCO1(Optimizer):


    def __init__(self, problem, epoch=10000, pop_size=100, sp=0.8, alpha=1.5, mp=0.5, **kwargs):

        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.sp = self.validator.check_float("sp", sp, (0, 1))
        self.alpha = self.validator.check_float("alpha", alpha, (1, 2))
        self.mp = self.validator.check_float("mp", mp, (0, 1))

        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False

    def perlin(self, current_position):
        perlin_values = []
        for i in range(self.problem.n_dims):
            perlin_values.append(
                pnoise1(current_position[i]))  # Use pnoise2() or pnoise3() for 2D or 3D noise respectively
        return perlin_values

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
        for idx in range(0, self.pop_size):
            #idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)

            pop_sorted = self.get_sorted_strim_population(self.pop)

            centroid = (self.g_best[self.ID_POS] + self.pop[idx_list[0]][self.ID_POS] + self.pop[idx_list[1]][self.ID_POS]) / 3
            centroid = (pop_sorted[0][self.ID_POS] + pop_sorted[1][self.ID_POS] + pop_sorted[2][self.ID_POS]) / 3
            #centroid = (pop_sorted[0][self.ID_POS] + pop_sorted[1][self.ID_POS] + self.pop[idx_list[0]][self.ID_POS]) / 3

            #pos_new = self.pop[idx][self.ID_POS] + self.levy_step() * (centroid - self.pop[idx][self.ID_POS])
            pos_new = self.pop[idx][self.ID_POS] + self.levy_step() * (centroid - self.pop[idx][self.ID_POS])
            self.cr = 0.7
            #self.cr = 0.0
            self.cr = 0.6
            #pnoise3()

            pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)

            pop.append([pos_new, None])

        pop = self.update_target_wrapper_population(pop)

        # create new pop by comparing fitness of corresponding each member in pop and children
        self.pop = self.greedy_selection_population(self.pop, pop)


    def levy_step(self):
        alpha = 1.5#1.5  # Parameter controlling the tail heaviness (typically > 1)
        sigma = (math.gamma(1 + alpha) * np.sin(math.pi * alpha / 2) / (
                    math.gamma((1 + alpha) / 2) * alpha * (2 ** ((alpha - 1) / 2)))) ** (1 / alpha)

        # Generate a step length
        u = np.random.uniform(0,1)  # Random number from a uniform distribution [0, 1]
        v = np.random.uniform(0, 1)  # Random number from a uniform distribution [0, 1]

        return (u / abs(v) ** (1 / alpha)) * sigma


class TCO2(Optimizer):

    def __init__(self, problem, epoch=10000, pop_size=100, wf=0.8, cr=0.9, strategy=0, **kwargs):
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.wf = self.validator.check_float("wf", wf, (0, 1.0))
        self.cr = self.validator.check_float("cr", cr, (0, 1.0))
        self.strategy = self.validator.check_int("strategy", strategy, [0, 5])

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
        for idx in range(0, self.pop_size):
            # idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 2, replace=False)
            idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)

            pop_sorted = self.get_sorted_strim_population(self.pop)


            centroid = (pop_sorted[0][self.ID_POS] + pop_sorted[1][self.ID_POS] + self.pop[idx][self.ID_POS] ) / 3
            centroid = (pop_sorted[0][self.ID_POS] + pop_sorted[1][self.ID_POS] +pop_sorted[2][self.ID_POS]) / 3

            #centroid2 = (self.pop[idx_list[2]][self.ID_POS]  + self.pop[idx_list[0]][self.ID_POS] + pop_sorted[2][self.ID_POS] ) / 3
           # centroid2 = (self.pop[idx_list[2]][self.ID_POS]  + self.pop[idx_list[0]][self.ID_POS] + pop_sorted[self.pop_size//2][self.ID_POS] ) / 3
            centroid2 = (self.pop[idx][self.ID_POS]  + self.pop[idx_list[0]][self.ID_POS] + self.pop[idx_list[1]][self.ID_POS] ) / 3
            centroid2 = (self.pop[idx_list[2]][self.ID_POS]  + self.pop[idx_list[0]][self.ID_POS] + self.pop[idx_list[1]][self.ID_POS] ) / 3


            #centroid2 = (self.pop[idx_list[2]][self.ID_POS]  + self.pop[idx_list[0]][self.ID_POS] + self.create_solution(self.problem.lb, self.problem.ub)[self.ID_POS] ) / 3

            # centroid = (pop_sorted[0][self.ID_POS] + pop_sorted[1][self.ID_POS] + self.pop[idx_list[0]][self.ID_POS]) / 3

            # pos_new = self.pop[idx][self.ID_POS] + self.levy_step() * (centroid - self.pop[idx][self.ID_POS])

            #self.cr = 0.5
            #self.cr = 0.6
            #self.cr = 0.4
            self.cr = 0.8
            #print("-------" + str(self.perlin(centroid)))
            if np.random.uniform(0, 1) < self.cr:
                pos_new = self.pop[idx][self.ID_POS] + np.random.randint(2, size=self.problem.n_dims) *  self.levy_step() * (centroid  - self.pop[idx][self.ID_POS])
                #pos_new = self.pop[idx][self.ID_POS] +  self.levy_step() *np.random.uniform(0.0, 1, self.problem.n_dims) * (centroid - self.pop[idx][self.ID_POS])

            else:
                #pos_new = self.pop[idx][self.ID_POS] +   np.multiply(np.random.randint(2, size=self.problem.n_dims), self.levy_step() * (centroid2 + self.perlin(centroid) * np.random.uniform(0.0, 1, self.problem.n_dims) - self.pop[idx][self.ID_POS]))
                #pos_new = self.pop[idx][self.ID_POS] +    self.perlin(centroid) * np.random.uniform(0.0, 1, self.problem.n_dims) * (centroid2 - self.pop[idx][self.ID_POS]) *
                pos_new = self.pop[idx][self.ID_POS] +  self.perlin(self.pop[idx][self.ID_POS]) * (centroid2 - self.pop[idx][self.ID_POS])

            '''
            if np.random.uniform(0, 1) < self.cr:
                pos_new = self.pop[idx][self.ID_POS] +  self.levy_step() * (centroid + self.perlin(centroid) * np.random.uniform(0.0, 1, self.problem.n_dims) - self.pop[idx][self.ID_POS])
            else:
                pos_new = self.pop[idx][self.ID_POS] +    self.levy_step() * (centroid2 + self.perlin(centroid) * np.random.uniform(0.0, 1, self.problem.n_dims) - self.pop[idx][self.ID_POS])
            '''

            #print(self.perlin(pos_new))
            #print(pos_new)
            #pos_new = pos_new + self.perlin(pos_new) * np.random.uniform(0.0, 2, self.problem.n_dims)

            #pos_new = self.mutation_process(pos_new)

            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            # self.cr = 0.0
            # pnoise3()
            #pos_new = self._mutation__(self.pop[idx][self.ID_POS], pos_new)
            pop.append([pos_new, None])

        pop = self.update_target_wrapper_population(pop)
        self.pop = self.greedy_selection_population(self.pop, pop)


    def levy_step(self):
        alpha = 1.5  # 1.5  # Parameter controlling the tail heaviness (typically > 1)
        sigma = (math.gamma(1 + alpha) * np.sin(math.pi * alpha / 2) / (
                math.gamma((1 + alpha) / 2) * alpha * (2 ** ((alpha - 1) / 2)))) ** (1 / alpha)

        # Generate a step length
        u = np.random.uniform(0, 1)  # Random number from a uniform distribution [0, 1]
        v = np.random.uniform(0, 1)  # Random number from a uniform distribution [0, 1]

        return (u / abs(v) ** (1 / alpha)) * sigma

    def perlin(self, current_position):
        perlin_values = []
        for i in range(self.problem.n_dims):
            perlin_values.append(
                pnoise1(current_position[i]))  # Use pnoise2() or pnoise3() for 2D or 3D noise respectively
        return perlin_values

    def select(self, pop_old_sorted, pop_new):
        keep = pop_old_sorted[:3]
        #for i
        the_pop = keep + self.get_sorted_strim_population(self.pop)


class TCO3(Optimizer):


    def __init__(self, problem, epoch=10000, pop_size=100, sp=0.8, **kwargs):
        super().__init__(problem, kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [10, 10000])
        self.sp = self.validator.check_float("sp", sp, (0, 1.0))
        self.nfe_per_epoch = self.pop_size
        self.sort_flag = False


    def evolve(self, epoch):
        """
        The main operations (equations) of algorithm. Inherit from Optimizer class

        Args:
            epoch (int): The current iteration
        """
        pop_sorted = self.get_sorted_strim_population(self.pop)

        pop = []
        for idx in range(0, self.pop_size):
            idx_list = np.random.choice(list(set(range(0, self.pop_size)) - {idx}), 3, replace=False)

            centroid = (pop_sorted[0][self.ID_POS] + pop_sorted[1][self.ID_POS] +pop_sorted[2][self.ID_POS]) / 3
            centroid2 = (self.pop[idx_list[2]][self.ID_POS]  + self.pop[idx_list[0]][self.ID_POS] + self.pop[idx_list[1]][self.ID_POS] ) / 3

            if np.random.uniform(0, 1) < self.sp:
                pos_new = self.pop[idx][self.ID_POS] + np.random.binomial(n=1, p=0.6, size=[self.problem.n_dims]) * self.levy_step()  * (centroid  - self.pop[idx][self.ID_POS])
            else:
                pos_new = self.pop[idx][self.ID_POS] + self.perlin(self.pop[idx][self.ID_POS]) * (centroid2 - self.pop[idx][self.ID_POS])
            pos_new = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop.append([pos_new, None])

        pop = self.update_target_wrapper_population(pop)

        self.pop = self.greedy_selection_population(self.pop, pop)


    def levy_step(self):
        alpha = 1.2  # 1.5  # Parameter controlling the tail heaviness (typically > 1)
        sigma = (math.gamma(1 + alpha) * np.sin(math.pi * alpha / 2) / (
                math.gamma((1 + alpha) / 2) * alpha * (2 ** ((alpha - 1) / 2)))) ** (1 / alpha)

        # Generate a step length
        u = np.random.uniform(0, 1)  # Random number from a uniform distribution [0, 1]
        v = np.random.uniform(0, 1)  # Random number from a uniform distribution [0, 1]

        return (u / abs(v) ** (1 / alpha)) * sigma

    def perlin(self, current_position):
        perlin_values = []
        for i in range(self.problem.n_dims):
            perlin_values.append(
                pnoise1(current_position[i]))
        return perlin_values

    def random_levy_step(self, alpha):
        # Generate a random step from the Cauchy distribution
        step = np.random.standard_cauchy()

        # Apply transformation to map the step to the range [0, 1]
        transformed_step = 0.5 + np.arctan(step) / np.pi

        return transformed_step
    def random_levy_step2(self):
        alpha = 1.5  # Shape parameter for the Beta distribution
        beta = 1.5   # Shape parameter for the Beta distribution

        # Generate a random step from the Beta distribution
        step = np.random.beta(alpha, beta)

        return step