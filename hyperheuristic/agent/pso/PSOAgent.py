import asyncio

from pyswarms.backend import compute_objective_function, compute_pbest
from pyswarms.backend.topology import Ring, Star
from pyswarms.single import GeneralOptimizerPSO

from hyperheuristic.model.ActualSolution import ActualSolution
# Import standard library
import logging

# Import modules
import numpy as np
import multiprocessing as mp

from collections import deque

from hyperheuristic.agent.pso.StarTopology import StarTopology


class PSOAgent(GeneralOptimizerPSO):
    pass

    id = 0

    def __init__(
        self,
        id,
        n_particles,
        dimensions,
        options,
        bounds=None,
        oh_strategy=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        ftol_iter=1,
        init_pos=None,
        topology=Ring(),
    ):
        GeneralOptimizerPSO.__init__(self, n_particles, dimensions, options, topology, bounds, oh_strategy, bh_strategy, velocity_clamp, vh_strategy, center, ftol, ftol_iter, init_pos)
        self.id = id
        #self.top = StarTopology()
        self.top = Star()

    def set_initial_particles_state(self, initial_particles):
        if initial_particles is not None:
            for idx, p in enumerate(initial_particles):
                self.swarm.position[idx] = p.position
                self.swarm.velocity[idx] = p.velocity
                self.swarm.pbest_pos[idx] = p.pbest_pos
                self.swarm.pbest_cost = np.append(self.swarm.pbest_cost, p.pbest_cost)

    def set_global_particle_state(self, particle):
        if particle is not None:
            self.swarm.best_cost = particle.value
            self.swarm.best_pos = particle.position

    def optimize(
            self, objective_func, iters, n_processes=None, verbose=False, **kwargs
    ):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """

        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} iters with {}".format(iters, self.options),
            lvl=log_level,
        )
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        ftol_history = deque(maxlen=self.ftol_iter)

        particles_per_iteration = {}
        best_values_per_iteration = []

        for i in self.rep.pbar(iters, self.name) if verbose else range(iters):
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=pool, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)

            ###### Retain costs per iterations
            particles_per_iteration[i] = to_particles(self, self.swarm, self.id)
            ######

            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(
                self.swarm, **self.options
            )
            #asyncio.run(self.exchanger.collect(self.swarm.best_pos, i))

            best_values_per_iteration.append(self.swarm.best_cost)
            # fmt: on
            if verbose:
                self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                    np.abs(self.swarm.best_cost - best_cost_yet_found)
                    < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform options update
            self.swarm.options = self.oh(
                self.options, iternow=i, itermax=iters
            )

            #global_best = asyncio.run(self.exchanger.obtain(iteration=i))

            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds#, global_best
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )

        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()

        final_best_particle = ActualSolution(self.id, final_best_cost, None, final_best_pos, None, None)
        return (particles_per_iteration, best_values_per_iteration, final_best_particle)


def to_particles(self, swarm, id):
    costs_touples = []
    for idx, cost in enumerate(swarm.current_cost):
        costs_touples.append(ActualSolution(id, cost, swarm.position[idx], swarm.velocity[idx], swarm.pbest_cost[idx], swarm.pbest_pos[idx]))
    return costs_touples