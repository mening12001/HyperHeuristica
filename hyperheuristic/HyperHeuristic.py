import abc


class HyperHeuristic(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def optimize(self, objective_func, dimensions, bounds, nr_hypergenerations=40):
        pass
