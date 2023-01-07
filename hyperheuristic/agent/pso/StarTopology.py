from pyswarms.backend import BoundaryHandler, VelocityHandler
from pyswarms.backend.topology import Star

from hyperheuristic.agent.pso import operators


class StarTopology(Star):
    pass


    def compute_velocity(
        self,
        swarm,
        clamp=None,
        vh=VelocityHandler(strategy="unmodified"),
        bounds=None,
        global_solutions=None
    ):
        """Compute the velocity matrix

        This method updates the velocity matrix using the best and current
        positions of the swarm. The velocity matrix is computed using the
        cognitive and social terms of the swarm.

        A sample usage can be seen with the following:

        .. code-block :: python

            import pyswarms.backend as P
            from pyswarms.backend.swarm import Swarm
            from pyswarms.backend.handlers import VelocityHandler
            from pyswarms.backend.topology import Star

            my_swarm = P.create_swarm(n_particles, dimensions)
            my_topology = Star()
            my_vh = VelocityHandler(strategy="adjust")

            for i in range(iters):
                # Inside the for-loop
                my_swarm.velocity = my_topology.update_velocity(my_swarm, clamp, my_vh,
                bounds)

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        clamp : tuple of floats (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        vh : pyswarms.backend.handlers.VelocityHandler
            a VelocityHandler instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.

        Returns
        -------
        numpy.ndarray
            Updated velocity matrix
        """
        return operators.compute_velocity(swarm, clamp, vh, bounds=bounds, global_solutions=global_solutions)



    def compute_position(
        self, swarm, bounds=None, bh=BoundaryHandler(strategy="periodic")
    ):
        """Update the position matrix

        This method updates the position matrix given the current position and
        the velocity. If bounded, it waives updating the position.

        Parameters
        ----------
        swarm : pyswarms.backend.swarms.Swarm
            a Swarm instance
        bounds : tuple of :code:`np.ndarray` or list (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        bh : pyswarms.backend.handlers.BoundaryHandler
            a BoundaryHandler instance

        Returns
        -------
        numpy.ndarray
            New position-matrix
        """
        return operators.compute_position(swarm, bounds, bh)
